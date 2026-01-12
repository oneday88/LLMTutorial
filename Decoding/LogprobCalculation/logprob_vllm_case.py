import os
# Set environment variable to use ModelScope before importing vLLM
os.environ['VLLM_USE_MODELSCOPE'] = 'True'

# Now import and use vLLM
from vllm import LLM, SamplingParams

# Initialize LLM - it will now automatically download from ModelScope if needed
model_name = "Qwen/Qwen3-4B-Thinking-2507"
llm = LLM(model=model_name,
    tensor_parallel_size=1
        )

"""
Test basic vllm configuration and setting
"""
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, 
         repetition_penalty=1.1,  # Penalize repeated tokens
        max_tokens=512)
prompts = ["At first the quick brown fox jumped over"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)


"""
Log-prob calculation
"""
# 1 Define the sequence parts
prompt = "At first the quick brown fox jumped over"
target_sequence = "then the cat sat on the mat" # Note the leading space for correct tokenization
full_text = prompt + target_sequence

# 2 Define SamplingParams:
sampling_params = SamplingParams(
    temperature=0.0,  # Set to 0 for deterministic sampling (optional)
    max_tokens=1,     # Set to a minimum value since we only want the prompt's logprob
    logprobs=1,       # Crucial: enables token logprob tracking
)

outputs = llm.generate([full_text], params)
# 4. Extract and calculate the total logprob
logprob_list = []
output = outputs[0]

# The logprobs for the input tokens are stored in the 'prompt_logprobs' list
if output.prompt_logprobs:
    # Iterate through the dictionary of logprobs for each token in the prompt
    for token_logprob_dict in output.prompt_logprobs:
        
        # 🔑 FIX: Check if the token_logprob_dict is None before trying to call .values()
        if token_logprob_dict is None:
            # This usually corresponds to the first token, which has no preceding context.
            logprob_list.append(0)
            continue 
        
        sub_record = next(iter(token_logprob_dict.values()))
        sub_logprob = sub_record.logprob
        logprob_list.append(sub_logprob)

print(f"Prompt: '{prompt}'")
print(f"Total Log Probability of the Prompt: {logprob_list}")


