import json
from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view

import torch
import matplotlib.pyplot as plt
import numpy as np

def map_words_to_token_ids(subwords):
    """
    example input:
        ['<s>', '▁In', '▁this', '▁hypoth', 'et', 'ical', '▁story', ',' ..]
    example output:
        [('<s>', [0]), ('In', [1]), ('this', [2]), ('hypothetical', [3, 4, 5]), ('story,', [6, 7], ...)
    """
    current_word = ""
    current_ids = []
    result = []

    for i, token in enumerate(subwords):
        if token.startswith("▁"):
            if current_word:  # about to start new one
                result.append((current_word, current_ids))
            current_word = token[1:]  # Remove the "▁" to get the real word
            current_ids = [i]
        else:
            current_word += token
            current_ids.append(i)

    # Add the last word to the dictionary
    if current_word:
        result.append((current_word, current_ids))

    return result


def eval_model_invariance_to_jailbreak_per_prompt_jailbreak_pair(prompt, jailbreaking_prompt, model, tokenizer):
    """
    lower is better
    """
    init_prompt_number_words = len(prompt.split())
    jailbreaking_inputs = tokenizer.encode(jailbreaking_prompt, return_tensors='pt')  # Tokenize input text
    jailbreaking_subwords = tokenizer.convert_ids_to_tokens(jailbreaking_inputs[0]) 
    # TODO @adib can you help with translating the line below for non-AIM jailbreaks?
    # this assumes that the task comes after the jailbreak in the final prompt
    orig_prompt_tokens_idxs_in_attn = [idx for entry in map_words_to_token_ids(jailbreaking_subwords)[-init_prompt_number_words:] for idx in entry[1] ]


    jailbreaking_outputs = model(jailbreaking_inputs)  
    jailbreaking_attention = jailbreaking_outputs[-1]  # Retrieve attention from model outputs


    candidate_metric = []
    for LAYER_IDX in range(len(jailbreaking_attention)):
        for HEAD_IDX in range(jailbreaking_attention[0].shape[1]):
            attending_matrix = torch.argsort(jailbreaking_attention[LAYER_IDX][0,HEAD_IDX,:,:], dim=1, descending=True)
            is_row_in_og_prompt = np.isin(np.arange(jailbreaking_attention[LAYER_IDX][0,HEAD_IDX,:,:].shape[0]), 
                                          orig_prompt_tokens_idxs_in_attn).reshape(-1, 1)
            is_val_in_og_prompt = np.isin(attending_matrix, orig_prompt_tokens_idxs_in_attn)
            
            # this new_arr seems fairly complex
            new_arr = np.where(is_row_in_og_prompt & is_val_in_og_prompt, 3, 
                           np.where(~is_row_in_og_prompt & is_val_in_og_prompt, 1, 
                           np.where(is_row_in_og_prompt & ~is_val_in_og_prompt, 2, 0)))

            # this is the simplest metric i can think of to turn new_arr into a single number
            # it adds up the ranks of non-jailbreak tokens in the final token's attention

            # for example, if the final token attention attends the most to a token in the o.g. task
            # and the second most to another token in the o.g. prompt, and so on
            # then this is minimized.

            # the more that final row in attention looks at jailbreak prompt, the higher this number gets
            # i think this assumes that final token is part of the original task TODO ileana extend this
            candidate_metric.append(np.sum(np.where(new_arr[-1] == 3)[0]))

    return np.mean(candidate_metric)


if __name__ == "__main__":
    base_model = AutoModel.from_pretrained("./Llama-2-7b-chat-hf", output_attentions=True)  # Configure model to return attention values
    prune_model = AutoModel.from_pretrained("./Llama-2-7b-chat-hf-20-sparsity", output_attentions=True)  # Configure model to return attention values
    tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf', cache_dir='llm_weights', use_fast=True)

    with open("malicious_task_maping_unstructured_30_llama-2.json") as f:
        data = json.load(f)
    result = {}

    # note that this could be batched for performance
    jailbreak_method = 'AIM'
    category_data = data[jailbreak_method]
    with torch.inference_mode():
        counter_1 = 1
        for category_name, subcategory_data in category_data.items():
            print(f"  category {counter_1} out of {len(category_data) + 1}")
            counter_2 = 1
            for task_name, task_data in subcategory_data.items():
                print(f"    task {counter_2} out of {len(subcategory_data) + 1}")
                counter_3 = 1
                for severity_name, examples in task_data.items():
                    for example_idx, prompts in enumerate(examples):
                        prompt = prompts["task"]
                        jailbreak_prompt = prompts["jailbreaking_prompt"]

                        val1 = eval_model_invariance_to_jailbreak_per_prompt_jailbreak_pair(prompt, jailbreak_prompt, base_model, tokenizer)
                        val2 = eval_model_invariance_to_jailbreak_per_prompt_jailbreak_pair(prompt, jailbreak_prompt, prune_model, tokenizer)
                        prune_improvement = val2 - val1
                        print(prune_improvement)

                    counter_3 += 1
                counter_2 += 1
            counter_1 += 1




