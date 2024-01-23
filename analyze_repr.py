import pickle
from transformers import AutoTokenizer, LlamaForCausalLM
import time
import torch
import argparse
import yaml
import json
import os
from transformers import GenerationConfig

def get_repr_prompt(text):
    return  f"The following text: '{text}' means in one word: "


def get_repr(model, tokenizer, prompt, device):
    llama2_prompt = f"[INST] {prompt} [\INST]"
    repr_prompt = get_repr_prompt(llama2_prompt)
    tokens = tokenizer(repr_prompt, return_tensors="pt").to(device)
    generation_config = GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=0)
    output = model.generate(tokens.input_ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True, generation_config=generation_config)
    prompt_repr = output['scores'][0]
    return prompt_repr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    args = parser.parse_args()

    device = torch.device("mps")
    model = LlamaForCausalLM.from_pretrained(args.model, cache_dir='llm_weights').to(device)
    tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf', cache_dir='llm_weights', use_fast=True)


    with open("malicious_task_maping_unstructured_30_llama-2.json") as f:
        data = json.load(f)

    result = {}

    # note that this could be batched for performance
    with torch.inference_mode():
        counter_1 = 1
        for jailbreak_method, category_data in data.items():
            print(f"jailbreak method {counter_1} out of {len(data) + 1}")
            result[jailbreak_method] = {}
            counter_2 = 1
            for category_name, subcategory_data in category_data.items():
                print(f"  category {counter_2} out of {len(category_data) + 1}")
                result[jailbreak_method][category_name] = {}
                counter_3 = 1
                for task_name, task_data in subcategory_data.items():
                    print(f"    task {counter_3} out of {len(subcategory_data) + 1}")
                    result[jailbreak_method][category_name][task_name] = {}
                    counter_4 = 1
                    for severity_name, examples in task_data.items():
                        result[jailbreak_method][category_name][task_name][severity_name] = []
                        for example_idx, prompts in enumerate(examples):
                            reference = prompts["task"]
                            adversarial = prompts["jailbreaking_prompt"]

                            reference_repr = get_repr(model, tokenizer, reference, device)
                            adversarial_repr = get_repr(model, tokenizer, adversarial, device)

                            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                            output = cos(reference_repr, adversarial_repr)

                            result[jailbreak_method][category_name][task_name][severity_name].append(output.item())
                        counter_4 += 1
                    counter_3 += 1
                counter_2 += 1
            counter_1 += 1

    # pickle is bad practice but this will do for now
    with open('data.pkl', 'wb') as file:
        pickle.dump(result, file)



if __name__=="__main__":
    main()
