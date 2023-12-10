# from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from datasets import load_dataset
from torch import nn
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
# model config
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

if __name__=="__main__":
    main()
