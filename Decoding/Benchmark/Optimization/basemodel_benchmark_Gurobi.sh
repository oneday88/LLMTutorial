#!/bin/bash
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PYTHONMULTIPROCESSING_SPAWN=1

python3 new_benchmark_eval.py --model_name "Qwen/Qwen3-4B-Instruct-2507"
python3 new_benchmark_eval.py --model_name "Qwen/Qwen3-4B-Thinking-2507"

