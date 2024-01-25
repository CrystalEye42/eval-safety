import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import torch


DEVICE = torch.device("mps")
model = AutoModelForCausalLM.from_pretrained(
        "./Llama-2-7b-chat-hf", output_attentions=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(
        'NousResearch/Llama-2-7b-chat-hf', cache_dir='llm_weights', use_fast=True)

with open("malicious_task_maping_unstructured_30_llama-2.json") as f:
    data = json.load(f)


def get_ppl(prompt, model, tokenizer):
    with torch.inference_mode():
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
        outputs = model(inputs)
        logits = outputs['logits']
        loss = torch.nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size),
                                                 inputs.view(-1))
        perplexity = torch.math.exp(loss)
        return {'perplexity': perplexity, 'num_tokens': inputs.shape[1]}


result = {}
for jailbreak_method, category_data in data.items():
    print(f"jailbreak method = {jailbreak_method}", flush=True)
    result[jailbreak_method] = {}
    for category_name, subcategory_data in category_data.items():
        result[jailbreak_method][category_name] = {}
        for task_name, task_data in subcategory_data.items():
            result[jailbreak_method][category_name][task_name] = {}
            for severity_name, examples in task_data.items():
                result[jailbreak_method][category_name][task_name][severity_name] = []
                for example_idx, prompts in enumerate(examples):
                    prompt = prompts["jailbreaking_prompt"]
                    ppl_result = get_ppl(prompt, model, tokenizer)
                    result[jailbreak_method][category_name][
                            task_name][severity_name].append(ppl_result)
# pickle is bad practice but this will do for now
with open('ppl.pkl', 'wb') as file:
    pickle.dump(result, file)
