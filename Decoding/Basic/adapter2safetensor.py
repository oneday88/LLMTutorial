import os
os.environ['VLLM_USE_MODELSCOPE'] = 'true'
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def adapter_to_safetensor(model_path, ckpt_path, output_path):
    state_dict = defaultdict(list)
    world_size = 8   # 8卡
    for rank in range(world_size):
        filepath = f"{ckpt_path}/model_world_size_{world_size}_rank_{rank}.pt"
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

# 从本地路径加载模型和分词器
local_model_path = "/DATA/disk1/chenyitian/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507"
# Load your Qwen3 base and the trained adapter
ckpt_path = "/DATA/disk2/chenyitian/checkpoints/Qwen3Distill/20251111/global_step_400/actor"
output_path = "./safetensor_ckpt"
adapter_to_safetensor(local_model_path, ckpt_path, output_path)
