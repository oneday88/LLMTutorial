"""
chenyitian@shanshu.ai
Task: to get the log-probability (log-likelihood) of any sequence of tokens using the **Hugging Face Transformers** library. 
The key is to use a single **forward pass** and then calculate the conditional log-probabilities using the returned logits.
This is a very common task in LLM evaluation, especially for tasks like Perplexity calculation, Reinforcement Learning (RLHF), and sequence scoring.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from modelscope import AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts import PromptTemplate


from utils import load_jsonl
from benchmark_prompt_utils import benchmark_gurobi_prompts

model_name = "Qwen/Qwen3-4B-Thinking-2507" # Example model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set padding token for batch processing if needed (good practice)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

"""
Load the dataset
"""
filepath = 'data/api_return/response_deepseek-chat_gurobi_IndustryOR_fixedV2.json'
### The testing dataset
loaded_data = load_jsonl(filepath)
if loaded_data:
    pass
    print(f"Successfully loaded {len(loaded_data)} items from {filepath}")

test_data =  loaded_data[:5]

### The prompt
prompt_name='zeroshot_q2mc_v2'
zeroshot_prompt =  benchmark_gurobi_prompts[prompt_name]
zeroshot_prompt = PromptTemplate.from_template(zeroshot_prompt)

sub_item = test_data[0]
sub_question = sub_item['en_question']
sub_question_str = zeroshot_prompt.format(Question=sub_question).strip()

sub_response_str = sub_item['response']

prompt, target_sequence = sub_question_str, sub_response_str

# 1 Define the sequence parts
#prompt = "At first the quick brown fox jumped over"
#target_sequence = "then the cat sat on the mat" # Note the leading space for correct tokenization
full_text = prompt + target_sequence

# 2\. Tokenization and Forward Pass: The process involves tokenizing the *full sequence* and feeding it into the model in a single forward pass.
# Tokenize the full sequence
encoded_full = tokenizer(full_text, return_tensors="pt")
input_ids = encoded_full['input_ids']
attention_mask = encoded_full['attention_mask']

# The prompt length is needed to calculate the conditional log-probabilities
prompt_len = tokenizer(prompt, return_tensors="pt")['input_ids'].shape[-1]
target_len = input_ids.shape[-1] - prompt_len

# 2. Perform a single forward pass
# The model is designed to return logits (raw prediction scores)
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

"""
### 3\. LogProb Calculation (The Chain Rule)
We use the chain rule: $\log P(S) = \sum_{i=1}^{N} \log P(t_i | t_1, \dots, t_{i-1})$.

1.  Convert logits to **log-probabilities** using `log_softmax`.
2.  **Shift Logits and Labels:** The logits at position $i$ predict token $t_{i+1}$. To align them, we compare the logits $t_{1..N-1}$ with the actual tokens $t_{2..N}$.
3.  **Extract Conditional LogProbs:** Select the log-probability corresponding to the actual token ID at each position.
4.  **Sum:** Sum the log-probabilities to get the sequence score.
"""
# 3. Apply log_softmax to get log probabilities
log_probs = F.log_softmax(logits, dim=-1)
target_token_ids = input_ids[:, prompt_len:]
conditional_log_probs = log_probs[:, prompt_len - 1 : -1, :]
# Select the log-probability of the actual token that occurred
# This uses torch.gather to select the log-prob for each token in the target_token_ids
sequence_log_probs = torch.gather(
    conditional_log_probs,
    2, # Dimension to gather from (the vocabulary dimension)
    target_token_ids.unsqueeze(-1)
).squeeze(-1)

sequence_log_likelihood = sequence_log_probs.sum(dim=-1).item()

# Print results
print(f"Prompt: '{prompt}'")
print(f"Target Sequence: '{target_sequence.strip()}'")
print(f"Token IDs of Target: {target_token_ids.tolist()[0]}")
print(f"Conditional LogProbs (per token): {sequence_log_probs.tolist()[0]}")
print(f"Total Log-Probability (P(target|prompt)): {sequence_log_likelihood:.4f}")
print(f"Total Probability (exp(LogProb)): {torch.exp(torch.tensor(sequence_log_likelihood)):.8f}")
