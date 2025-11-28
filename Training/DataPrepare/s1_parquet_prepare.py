"""
Data prepare for online policy distllation with verl
"""
import os
from datetime import datetime, date
import argparse
import datasets #hugging face de datesets package
from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs
from langchain.prompts import PromptTemplate

from utils import load_jsonl,fix_data_types
from rule_prompt_utils import benchmark_gurobi_prompts


"""
if __name__ == '__main__':
"""
parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='/home/chenyitian/PolicyDistillation/data/api_return')#数据加载路径
parser.add_argument("--data_name",default = ['response_deepseek-chat_gurobi_IndustryOR_fixedV2.json', 'response_deepseek-chat_gurobi_OptMATH_Bench_166.json','response_deepseek-chat_gurobi_MAMO_ComplexLP_fixed.json']) #test_data
parser.add_argument('--output_path', default='/home/chenyitian/PolicyDistillation/data/rldata')# Output path
parser.add_argument('--output_name',type=str, required=False, default='onlineDistll') # Output path
 
parser.add_argument('--prompt_type', type=str, required=False, default='user', choices=['user','system'], help='prompt type, options: []')
parser.add_argument('--prompt_name', type=str, required=False, default='zeroshot_q2mc_v2', help='prompt name, options: []')
parser.add_argument('--solver_name', type=str, required=False, default='gurobi',choices=['gurobi','copt','math'], help='solver name, options: []')

args = parser.parse_args()

data_path = args.data_path              # The path of the testing dataset
data_name = args.data_name              # The testing data, in json format
solver_name = args.solver_name

prompt_type = args.prompt_type
output_path = args.output_path
output_name = args.output_name
os.makedirs(output_path, exist_ok=True)

### The prompt template: gurobi, copt, math

if solver_name == "gurobi":
    zeroshot_prompt_system = PromptTemplate.from_template( benchmark_gurobi_prompts[args.prompt_name]['system'])
    zeroshot_prompt_user = PromptTemplate.from_template( benchmark_gurobi_prompts[args.prompt_name]['user'])
    ability = 'or'
elif solver_name == "copt":
    zeroshot_prompt_system = PromptTemplate.from_template(benchmark_copt_prompts[args.prompt_name]['system'])
    zeroshot_prompt_user = PromptTemplate.from_template(benchmark_copt_prompts[args.prompt_name]['user'])
    ability = 'or'
elif solver_name == 'math':
    zeroshot_prompt_system = PromptTemplate.from_template(benchmark_math_prompts[args.prompt_name]['system'])
    zeroshot_prompt_user = PromptTemplate.from_template(benchmark_math_prompts[args.prompt_name]['user'])
    ability = 'math'

def make_map_fn():
    def process_fn(example):
        if prompt_type == 'user':
            prompt = [{"role": "user","content": zeroshot_prompt_user.format(Question=example['en_question']).strip() }]
        else:
            prompt = [{"role": "system", "content": zeroshot_prompt_system.format(Question=example['en_question']).strip() },
            {"role": "user","content": zeroshot_prompt_user.format(question=example['en_question']).strip() }]

        data = {
            "data_source": output_name,
            "prompt":prompt, 
            "ability": ability,
            "reward_model": {
                    "ground_truth": example['en_answer']
            },
            "extra_info": {'index':example['response']}
        }
        return data
    return process_fn

"""
Load the dataset: multiple files in the fold
"""

test_data = []
for sub_name in data_name[:]:
    sub_data_path = os.path.join(args.data_path,sub_name)
    sub_data = load_jsonl(sub_data_path)
    test_data.extend(sub_data)

test_data_fixed = fix_data_types(test_data)
dataset_dict = Dataset.from_list(test_data_fixed)

train_dataset = dataset_dict.map(function=make_map_fn())
test_dataset = dataset_dict.map(function=make_map_fn())
# print an example
print(train_dataset[1])

# Method 1: Using strftime
current_date = datetime.now()
yyyymmdd_str = current_date.strftime("%Y%m%d")

train_dataset.to_parquet(os.path.join(output_path, prompt_type + '_'+ output_name+'_'+yyyymmdd_str +'_' + 'train.parquet'))
test_dataset.to_parquet(os.path.join(output_path, prompt_type +  '_'+ output_name+'_'+yyyymmdd_str +'_' + 'test.parquet'))
