# Safety in Pruning
This is a repository for replicating the experiments from our paper: [Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning
](https://arxiv.org/abs/2401.10862). 

# Getting Started
Install the dependencies and obtain a Wanda pruned model checkpoint as described in the [original repository](https://github.com/locuslab/wanda)

## Generating outputs for our jailbreaking dataset
Run the following command to generate model responses to our jailbreaking dataset (`integrated.yaml`). Depending on the base model, set the prompt template to be one of llama, vicuna, or mistral for correct inference. 

```
python inference.py \
  --model path/to/model \
  --dataset path/to/dataset \
  --template llama|vicuna|mistral
```

## Benchmarking model
We provide methods for running various benchmarks. To run the [AltQA](https://huggingface.co/datasets/abacusai/WikiQA-Altered_Numeric_QA) long context test or the [WikiText](https://huggingface.co/datasets/wikitext) perplexity test, run the following. Depending on the base model, set the prompt template to be one of `llama`, `vicuna`, or `mistral` for correct inference. 
```
python evaluate.py \
  --model_path path/to/model \
  --output_path path/to/output/directory \
  --template llama|vicuna|mistral \
  --benchmark altqa|wikitext
```
