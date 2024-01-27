import argparse
from ruamel.yaml import YAML
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import torch
yaml = YAML(typ='safe')
yaml.preserve_quotes = True
yaml.default_flow_style = False
yaml.allow_unicode = True


DEVICE = torch.device("mps")


def get_ppl(prompt, model, tokenizer):
    with torch.inference_mode():
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
        outputs = model(inputs[:, :-1])
        logits = outputs['logits']
        loss = torch.nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size),
                                                 inputs[:, 1:].view(-1))
        return {'loss': loss.item(), 'num_tokens': inputs.shape[1]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--save_to', type=str, help='Output File Name')
    parser.add_argument('--eval_malicious', action="store_true")
    parser.add_argument('--eval_benign', action="store_true")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(
            'NousResearch/Llama-2-7b-chat-hf', cache_dir='llm_weights', use_fast=True)

    result = {}
    if args.eval_malicious:
        with open("malicious_task_maping_unstructured_30_llama-2.json") as f:
            data = json.load(f)

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
    if args.eval_benign:
        with open("benign_integrated.yaml") as f:
            data = yaml.load(f)
        for jailbreak_method, examples in data.items():
            print(f"jailbreak method = {jailbreak_method}", flush=True)
            result[jailbreak_method] = []
            for prompt in examples:
                ppl_result = get_ppl(prompt, model, tokenizer)
                result[jailbreak_method].append(ppl_result)

    # pickle is bad practice but this will do for now
    with open(args.save_to, 'wb') as file:
        pickle.dump(result, file)
