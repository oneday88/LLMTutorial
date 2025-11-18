"""
pip install -U evaluate
"""
import evaluate
perplexity = evaluate.load("perplexity", module_type="metric")
input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]

results = perplexity.compute(model_id='Qwen/Qwen2.5-7B-Instruct',
                             add_start_token=False,
                             predictions=input_texts)
print(list(results.keys()))
~                                 
