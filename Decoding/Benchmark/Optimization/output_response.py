import os
import tempfile
import shutil
import torch
import copy
import torch.distributed
from torch.distributed import init_device_mesh
from verl.utils.distributed import initialize_global_process_group
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
import argparse
import time, re
import argparse
import json
import tiktoken
import subprocess
from copy import deepcopy
from openai import OpenAI
from collections import defaultdict
from langchain.prompts import PromptTemplate
import multiprocessing 
from content_utils import extract_code_block,extract_obj,change_variable_types
import numpy as np
from vllm import LLM, SamplingParams        
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq
from rule_prompt_utils import benchmark_gurobi_prompts2, gurobi_prompt_temp,optimind_prompt
from rule_prompt_utils_sft import copt_prompt_temp
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from utils import load_jsonl

def get_sampling_params(decoding_type):
    stop_tokens = ['<|im_end|>','</s>']
    if args.decoding_method == "greedy":
        return SamplingParams(
            n=args.topk,
            temperature=0,
            top_p=1,
            top_k=1,
            max_tokens=args.max_tokens,
            stop=stop_tokens,
            repetition_penalty=args.repetition_penalty
        )
    elif args.decoding_method == "sampling":
        return SamplingParams(
            n=args.topk,
            temperature=0.5,
            top_p=0.95,
            max_tokens=args.max_tokens,
            stop=stop_tokens,
            repetition_penalty=args.repetition_penalty
        )
    elif args.decoding_method == "min_p":
        return SamplingParams(
            n=args.topk,
            temperature=0.8,
            min_p=0.1,
            max_tokens=args.max_tokens,
            stop=stop_tokens,
            repetition_penalty=args.repetition_penalty
        
        )
    elif args.decoding_method == "beamsearch":
        return SamplingParams(
            n=args.topk,
            temperature=0,
            top_p=1,
            max_tokens=args.max_tokens,
            stop=stop_tokens,
            use_beam_search=True,
            best_of=args.beam_size
        )
    else:
        raise ValueError(f"Unsupported decoding method: {args.decoding_method}")

def model2code(llm_output):
    pattern = r'<model>(.*?)</model>'
    match = re.search(pattern, llm_output, re.DOTALL)
    obj = None
    if match:
        obj = match.group(1).strip()
    if args.prompt_type == "user":
        prompt = [{
        "role": "user",
        "content": zeroshot_prompt.format(question=obj).strip() }]
    else:
        prompt = [
        {
        "role": "system",
        "content": zeroshot_prompt_system.format(question=obj).strip() },
        {
        "role": "user",
        "content": zeroshot_prompt_user.format(question=obj).strip() }]
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return text

def hf_model():
    state_dict = defaultdict(list)
    world_size = 8   # 8卡
    for rank in range(world_size):
        filepath = f"{checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)
    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)

def generate_with_model(model, prompt, sampling_params,batch_size):   
    results = []
    for i in range(0, len(prompt), batch_size):
        batch = prompt[i:i + batch_size]
        response = model.generate(batch, sampling_params)
        batch_texts = [g.outputs[0].text for g in response]
        results.extend(batch_texts)
    return results

def mp_worker(item):
    sub_question = item['en_question' ]
    if args.prompt_type == "user":
        prompt = [{
        "role": "user",
        "content": zeroshot_prompt.format(question=item['en_question']).strip() }]
    else:
        prompt = [
        {
        "role": "system",
        "content": zeroshot_prompt_system.format(question=item['en_question']).strip() },
        {
        "role": "user",
        "content": zeroshot_prompt_user.format(question=item['en_question']).strip() }]
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    #result_str = generate_with_model(model, question_str, sampling_params)
    return text

def check_result(result_str, item, cri):
    sub_answer = item['en_answer']
    if sub_answer == "No Best Solution" or "-9999" in str(sub_answer):
        sub_answer = None
    else:
        sub_answer = float(item['en_answer'])
    code_snippet = extract_code_block(result_str,solver_name)
    if code_snippet is None:
        return 2
    try:
        result = subprocess.run(['python3', '-c', code_snippet], capture_output=True, text=True, timeout=100)
    except subprocess.TimeoutExpired as e:
        if sub_answer is None:
            return 1
        else:
            return 0
    if result.returncode !=0 :
        return 3
    solver_result = extract_obj(result.stdout)
    result_str = change_variable_types(result_str)
    if result_str:
        if (solver_result and sub_answer and (np.abs(solver_result-sub_answer)/(np.abs(sub_answer)+1)>1e-6)) or solver_result is None:
            code_snippet = extract_code_block(result_str,solver_name)
            try:
                result = subprocess.run(['python3', '-c', code_snippet], capture_output=True, text=True, timeout=100)
                if result.returncode == 0 and extract_obj(result.stdout) and sub_answer and np.abs(extract_obj(result.stdout)- sub_answer)/(np.abs(sub_answer)+1)<1e-6:
                    return 1
            except subprocess.TimeoutExpired:
                #print("over_time")
                if sub_answer is None:
                    return 1
                else:
                    return 0
    if sub_answer and solver_result:
        sub_answer2  = -999999
        if cri:       
            return int((np.abs(solver_result-sub_answer)<0.01) or (np.abs(solver_result-sub_answer2)<0.01))
        else:
            return int((np.abs(solver_result-sub_answer)/(np.abs(sub_answer)+1)<1e-6) or (np.abs(solver_result-sub_answer2)/(np.abs(sub_answer2)+1)<1e-6))
    elif sub_answer == solver_result:
        return 1
    elif 'nfeasible' in result.stdout:
        if sub_answer is None:
            return 1
        else:
            return 0
    else:
        return 3

if __name__ == '__main__':
    parser.add_argument('--tensor_parallel_size', type=int, required=False, default=4, help='tensor parallel size, options: []')
    parser.add_argument("--topk", type=int, default=1, help="Number of generations per prompt")
    parser.add_argument("--decoding_method", type=str, default="sampling", choices=["greedy", "sampling", "beamsearch", "min_p"],help="Decoding method")  
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum number of tokens to generate")
    parser.add_argument("--repetition_penalty", type=float, default=1.02, help="Repetitionpenalty")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument('--prompt_type', type=str, required=False, default='user', help='prompt name, options: []')
    args = parser.parse_args()

    model_path = args.model_name
    checkpoint_path = args.checkpointpath
    checkpoint_date = checkpoint_path.split('/')[-3]
    checkpoint_name = checkpoint_path.split('/')[-2]
    multiprocessing.set_start_method('spawn')
    solver_name = 'gurobi'
    batch_size = 256
    # 如果checkpoint下有文件直接加载，否则把adapter转成pretrained再加载
    if args.checkpoint:
        print("Testing transfer adapter to safetenor-format checkpoint")
        output_path = os.path.join(checkpoint_path,"pretrained")
        if not os.path.exists(output_path):
            hf_model()
        model = LLM(
            model=output_path,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.7)
    else:
        model = LLM(
            model=model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True)
    print("Model initialized.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    #加载数据
    data_name = args.data_name
    for name in data_name[2:5]:
        try:
            dataset = os.path.join(args.datapath,name)
            print('Loading data', dataset)
            loaded_data = load_jsonl(dataset)
            test_data = loaded_data
            print('Finish Loading')
        except:
            print("No data!")
            continue
        
        if solver_name == "gurobi":
            zeroshot_prompt_system = PromptTemplate.from_template(optimind_prompt['system'])
            zeroshot_prompt_user = PromptTemplate.from_template(optimind_prompt['user'])
        else:
            zeroshot_prompt_system = PromptTemplate.from_template(copt_prompt_temp['system1'])
            zeroshot_prompt_user = PromptTemplate.from_template(copt_prompt_temp['user1'])
        sampling_params = get_sampling_params(args)
       # generate Q-M-C 
        prompt_list = []
        for item in test_data:
            prompt_list.append(mp_worker(item))
        result_strs = generate_with_model(model, prompt_list, sampling_params,batch_size)
        
        snippet_package_abs = []
        for result_str,item in zip(result_strs,test_data):
            snippet_package_abs.append(check_result(result_str,item,False))
        
        result_info = {
            0 : "Error anwser or timeout, got score 0",
            1 : "Right, got score 1",
            2 : "No code, got score 2",
            3 : "Error code, got score 3"
        }
        name_prefix = name.split('.')[0]
        
        model_name = 
        dir_path1 = f'/home/chenyitian/RLVR/Eval/response/{checkpoint_date}/{checkpoint_name}'
        file_path1 = os.path.join(dir_path1,f'{name_prefix}_response.jsonl')
        os.makedirs(dir_path1, exist_ok=True)
        
        with open(file_path1,'w',encoding='utf-8') as f:
            for score,result_str in zip(snippet_package_abs,result_strs):
                result = {
                    "score" : score,
                    "result_info" : result_info[score],
                    "response" : result_str 
                }
                f.write(json.dumps(result) + '\n')
        

        result_count = np.bincount(snippet_package_abs)
        #print(result_count)
        #result1 = np.bincount(snippet_package_m2c)
        dir_path2 = f'/home/chenyitian/RLVR/Eval/pass@1/{checkpoint_date}'
        file_path2 = os.path.join(dir_path2,f'{checkpoint_name}_pass@1.txt')
        os.makedirs(dir_path2,exist_ok=True)
        with open(file_path2,'a',encoding='utf-8') as fp:
            fp.write(f'{name_prefix} result is {result_count}\n{name_prefix} pass@1 is {result_count[1]/sum(result_count)}\n\n')
