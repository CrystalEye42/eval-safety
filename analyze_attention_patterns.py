"""
**** Experimental plan ****
1. How much in-task tokens attend to in-task tokens
2. How much jailbreak tokens attend to jailbreak tokens
3. How much last token attends to in-task tokens
4. How sharp attention patterns are - entropy
"""

import json
from transformers import AutoTokenizer, AutoModel
import pickle
import torch
import numpy as np


DEVICE = torch.device("mps")

# JAILBREAK_METHOD -> Union[int, Tuple[int, int]]
# If a jailbreak method maps to an int, it means
# the jailbreak prompt looks like f"{jailbreak_pretext} {task}"
# and the corresponding value counts the number of tokens in jailbreak_pretext.
# If the jailbreak method maps to a tuple, it means
# that the jailbreak prompt looks like f"{jailbreak_pretext} {task} {jailbreak_pretext}"
# and the two ints in the tuple count the tokens in either range of jailbreak_pretext.
PROMPT_SPEC = {
    "AIM": 437,
    "DAN": 248,
    "PERSONGPT": 455,
    "SIM": 460,
    "KEVIN": 172,
    "TRANSLATOR": 334,
    "CHARACTER": (10, 49),
    "CODE": (87, 17),
    "GAME": (26, 99),
    "TOMNJERRY": (128, 39),
}


def get_orig_prompt_tokens_idx(jailbreak_subwords, jailbreak_method):
    """
    example jailbreak_subwords:
        ['<s>', '▁prefix', '▁j', 'ail', 'break', '▁pre', 'text', "▁'",
        '▁tell', '▁me', '▁how', '▁to', '▁do', '▁something', '▁evil',
        "▁'", '▁suffix', '▁j', 'ail', 'break', '▁pre', 'text']
    example output:
        [8, 9, 10, 11, 12, 13, 14]

    inclusive enumeration of indices in jailbreak_subwords s.t.
    corresponding entries for that index are in the task text
    """
    if jailbreak_method == "ORIGINAL":
        raise ValueError(
            "ORIGINAL is the noop jailbreak method. ",
            "You shouldn't be calling this function with ORIGINAL.",
        )
    if jailbreak_method not in PROMPT_SPEC:
        raise ValueError(f"Jailbreak method not supported. Got {jailbreak_method}.")
    prompt_spec = PROMPT_SPEC[jailbreak_method]
    if isinstance(prompt_spec, int):
        prompt_idxs = range(prompt_spec, len(jailbreak_subwords))
    if isinstance(prompt_spec, tuple):
        start = prompt_spec[0]
        end = len(jailbreak_subwords) - prompt_spec[1]
        prompt_idxs = range(start, end)

    return prompt_idxs


def is_last_token_in_jailbreak(jailbreak_method):
    prompt_spec = PROMPT_SPEC[jailbreak_method]
    return isinstance(prompt_spec, tuple)


def get_ranking_arr(
    jailbreak_attention, LAYER_IDX, HEAD_IDX, orig_prompt_tokens_idxs_in_attn
):
    """
    input is a list of N_LAYER tensors
    each entry has shape (1, N_HEAD, N_SEQ, N_SEQ)
    (the one comes from batch size = 1)

    output looks only at LAYER_IDX and HEAD_IDX
    # (N_SEQ, N_SEQ) shape
    # values are -1, 0, 1, 2, 3

    if result[i][j] = k it means that:
    token i attends jth most to a token s.t.

     * if k = 0: i and the rank j token in i's attention are both jailbreak tokens
     * if k = 0: i is a jailbreak token, the rank j token in i's attention is a task token
     * [...] and so on
    if k = -1 then that corresponds to one of the masked inputs, we ignore this part of result
    """
    attention_pattern = jailbreak_attention[LAYER_IDX][0, HEAD_IDX, :, :].cpu()

    # (N_SEQ, N_SEQ), entries are indices
    attending_matrix = torch.argsort(attention_pattern, dim=1, descending=True)

    n_seq = attention_pattern.shape[-1]
    is_row_in_og_prompt = np.isin(
        np.arange(n_seq), orig_prompt_tokens_idxs_in_attn
    ).reshape(-1, 1)
    is_val_in_og_prompt = np.isin(attending_matrix, orig_prompt_tokens_idxs_in_attn)

    ranking_arr = np.where(
        attention_pattern == 0,
        -1,
        np.where(
            is_row_in_og_prompt & is_val_in_og_prompt,
            3,
            np.where(
                ~is_row_in_og_prompt & is_val_in_og_prompt,
                1,
                np.where(is_row_in_og_prompt & ~is_val_in_og_prompt, 2, 0),
            ),
        ),
    )

    return ranking_arr


def metric1(jailbreak_attention, LAYER_IDX, HEAD_IDX, orig_prompt_tokens_idxs_in_attn):
    """
    the lower this is, the more in-task parts of the prompt attend to themselves
    (sum of ranks)
    """
    ranking_arr = get_ranking_arr(
        jailbreak_attention, LAYER_IDX, HEAD_IDX, orig_prompt_tokens_idxs_in_attn
    )
    return np.sum(np.where(ranking_arr == 3)[1])


def metric2(jailbreak_attention, LAYER_IDX, HEAD_IDX, orig_prompt_tokens_idxs_in_attn):
    """
    the lower this is, the more jailbreak parts of the prompt attend to themselves
    (sum of ranks)
    """
    ranking_arr = get_ranking_arr(
        jailbreak_attention, LAYER_IDX, HEAD_IDX, orig_prompt_tokens_idxs_in_attn
    )
    return np.sum(np.where(ranking_arr == 0)[1])


def metric3(
    jailbreak_attention,
    LAYER_IDX,
    HEAD_IDX,
    jailbreak_method,
    orig_prompt_tokens_idxs_in_attn,
):
    """
    the lower this is, the more the last token attends to task prompt tokens
    (and not jailbreak part). Computed as a sum of ranks.
    """
    ranking_arr = get_ranking_arr(
        jailbreak_attention, LAYER_IDX, HEAD_IDX, orig_prompt_tokens_idxs_in_attn
    )
    cond = ranking_arr[-1] == (1 if is_last_token_in_jailbreak(jailbreak_method) else 3)
    return np.sum(np.where(cond)[0])


def metric4(jailbreak_attention, LAYER_IDX, HEAD_IDX):
    """
    entropy of attn patterns
    """
    p = (
        jailbreak_attention[LAYER_IDX][0, HEAD_IDX, :, :].cpu().numpy()
    )  # (N_SEQ, N_SEQ)
    return np.nansum(-p * np.log2(p), axis=1)


def metric5(jailbreak_attention, LAYER_IDX, orig_prompt_tokens_idxs_in_attn):
    jailbreak_attention = jailbreak_attention[LAYER_IDX]
    total_attention = torch.sum(
        jailbreak_attention[:, :, orig_prompt_tokens_idxs_in_attn, :]
    )
    intra_task_attention = torch.sum(
        jailbreak_attention[
            :, :, orig_prompt_tokens_idxs_in_attn, orig_prompt_tokens_idxs_in_attn
        ]
    )
    ratio = intra_task_attention / total_attention
    ratio = ratio.cpu().item()
    return ratio


def eval_metric(jailbreak_prompt, model, tokenizer, jailbreak_method, metric_name):
    jailbreak_inputs = tokenizer.encode(jailbreak_prompt, return_tensors="pt").to(
        DEVICE
    )
    jailbreak_subwords = tokenizer.convert_ids_to_tokens(jailbreak_inputs[0])
    orig_prompt_tokens_idxs_in_attn = get_orig_prompt_tokens_idx(
        jailbreak_subwords, jailbreak_method
    )

    jailbreak_outputs = model(jailbreak_inputs)
    jailbreak_attention = jailbreak_outputs[-1]

    candidate_metric = []
    if metric_name == "metric5":
        for LAYER_IDX in range(len(jailbreak_attention)):
            metric5(jailbreak_attention, LAYER_IDX, orig_prompt_tokens_idxs_in_attn)
    else:
        for LAYER_IDX in range(len(jailbreak_attention)):
            for HEAD_IDX in range(jailbreak_attention[0].shape[1]):
                if metric_name == "metric4":
                    candidate_metric.append(
                        metric4(jailbreak_attention, LAYER_IDX, HEAD_IDX)
                    )
                if metric_name == "metric3":
                    candidate_metric.append(
                        metric3(
                            jailbreak_attention,
                            LAYER_IDX,
                            HEAD_IDX,
                            jailbreak_method,
                            orig_prompt_tokens_idxs_in_attn,
                        )
                    )
                if metric_name == "metric2":
                    candidate_metric.append(
                        metric2(
                            jailbreak_attention,
                            LAYER_IDX,
                            HEAD_IDX,
                            orig_prompt_tokens_idxs_in_attn,
                        )
                    )
                if metric_name == "metric1":
                    candidate_metric.append(
                        metric1(
                            jailbreak_attention,
                            LAYER_IDX,
                            HEAD_IDX,
                            orig_prompt_tokens_idxs_in_attn,
                        )
                    )

    return np.mean(candidate_metric)


def get_model(model_choice):
    if model_choice == "base":
        fn = "./Llama-2-7b-chat-hf"
    else:
        percentage = model_choice.split("_")[0]
        fn = f"./Llama-2-7b-chat-hf-{percentage}-sparsity"
    print(f"!!!! loading model from {fn}")
    model = AutoModel.from_pretrained(fn, output_attentions=True).to(DEVICE)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "10_sparse", "20_sparse", "30_sparse"],
        help="LLaMA model",
    )
    parser.add_argument("--save_to", type=str, help="Output File Name")
    args = parser.parse_args()

    model = get_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf", cache_dir="llm_weights", use_fast=True
    )

    with open("malicious_task_maping_unstructured_30_llama-2.json") as f:
        data = json.load(f)
    result = {}

    # note that this could be batched for performance
    result = {}
    with torch.inference_mode():
        for candidate_metric in ["metric5"]:
            print(f"****metric = {candidate_metric}", flush=True)
            for jailbreak_method, category_data in data.items():
                if jailbreak_method == "ORIGINAL":
                    continue
                print(f"**jailbreak method = {jailbreak_method}", flush=True)
                counter_1 = 1
                result[jailbreak_method] = {}
                for category_name, subcategory_data in category_data.items():
                    print(
                        f"  category {counter_1} out of {len(category_data) + 1}",
                        flush=True,
                    )
                    counter_2 = 1
                    result[jailbreak_method][category_name] = {}
                    for task_name, task_data in subcategory_data.items():
                        print(
                            f"    task {counter_2} out of {len(subcategory_data) + 1}",
                            flush=True,
                        )
                        counter_3 = 1
                        result[jailbreak_method][category_name][task_name] = {}
                        for severity_name, examples in task_data.items():
                            result[jailbreak_method][category_name][task_name][
                                severity_name
                            ] = []
                            for example_idx, prompts in enumerate(examples):
                                prompt = prompts["task"]
                                jailbreak_prompt = prompts["jailbreaking_prompt"]

                                val = eval_metric(
                                    jailbreak_prompt,
                                    model,
                                    tokenizer,
                                    jailbreak_method,
                                    candidate_metric,
                                )
                                result[jailbreak_method][category_name][task_name][
                                    severity_name
                                ].append(val)

                            counter_3 += 1
                        counter_2 += 1
                    counter_1 += 1
            # pickle is bad practice but this will do for now
            with open(args.save_to, "wb") as file:
                pickle.dump(result, file)
