#from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from datasets import load_dataset
from torch import nn
import tqdm
import re
#from lm_eval_adaptor import LMEvalAdaptor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument('--template', default='llama', choices=['llama', 'mistral', 'uncensored', 'system_llama', 'vicuna'], help='model name for prompt template to use')
parser.add_argument("--benchmark", default='wikitext', choices=['wikitext', 'altqa', 'openllm'], help='method to use')
parser.add_argument("--output_path", default=None, type=str)
# following args probably not necessary
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

access_token = os.environ['HF_TOKEN']

def main():
    model = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir='llm_weights').to(device)
    enc = AutoTokenizer.from_pretrained(args.model_path, cache_dir='llm_weights', use_fast=False, trust_remote_code=True)

    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = 2048
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []

    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    results = {"ppl": ppl.item()}
    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)

def benchmark():
    model = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir='llm_weights').to(device)
    enc = AutoTokenizer.from_pretrained(args.model_path, cache_dir='llm_weights', use_fast=False, trust_remote_code=True)
    task_names = args.tasks.split(",")
    tasks = {
        "arc_challenge": 25,
        "arc_easy": 25,
        "hellaswag": 5,
        "mmlu": 0,
        "truthfulqa": 0,
        "winogrande": 5,
        "gsm8k": 5
    }
    for task_name in task_names:
        num_fewshot = tasks[task_name]
        print("Evaluating", task_name)
        lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
        results = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=[task_name],
            batch_size=args.batch_size,
            no_cache=True,
            num_fewshot=num_fewshot,
        )
    
        print(evaluator.make_table(results))
    
        if args.output_path is not None:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # otherwise cannot save
            results["config"]["model"] = args.model_path
            with open(args.output_path, "a") as f:
                json.dump(results, f, indent=2)


def long_context():
    model = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir='llm_weights').to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir='llm_weights', use_fast=False, trust_remote_code=True)

    if model.config.max_position_embeddings > 2048:
        testenc = load_dataset("abacusai/WikiQA-Altered_Numeric_QA", split="4k", cache_dir='llm_weights')
        max_len = 4000
    else:
        testenc = load_dataset("abacusai/WikiQA-Altered_Numeric_QA", split="2k", cache_dir='llm_weights')
        max_len = 2000

    if args.template == 'llama' or args.template == 'mistral':
        orig_prompt = "[INST] {prompt}. Answer with just a number. [\INST]"
        delim = '[\INST]'
    elif args.template == 'uncensored':
        orig_prompt = "### HUMAN:\n {prompt}. Answer with just a number. \n\n### RESPONSE:\n"
        delim = 'RESPONSE:'
    elif args.template == 'vicuna':
        orig_prompt = "USER: {prompt}. Answer with just a number. \nASSISTANT: "
        delim = 'ASSISTANT:'
    orig_prompt = "[INST] {prompt}. Answer with just a number. [\INST]"

    correct = 0

    results = {'acc': 0, 'ground_truth': [], 'responses': []}

    for sample in tqdm.tqdm(testenc, desc="evaluating..."):
        inp_text = sample['conversations'][0]['value']
        prompt = orig_prompt.format(prompt=inp_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        generate_ids = model.generate(inputs.input_ids, max_length=3000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split(delim)[-1]
                        
        result = response.split("Inst]")[-1].strip()
        if correct == 0:
            print("Expected:", sample['conversations'][1]['value'], " Got:", result)
        results['ground_truth'].append(sample['conversations'][1]['value'])
        results['responses'].append(result)
        searched = re.search('\d+', result)

        if searched is not None and sample['conversations'][1]['value'] == searched.group(0):
            correct += 1
    results["acc"] = correct / len(testenc)

    print("Acc:", results["acc"])

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)

    
if __name__=="__main__":
    if args.benchmark == 'wikitext':
        main()
    elif args.benchmark == 'altqa':
        long_context()
    elif args.benchmark == 'openllm':
        benchmark()
