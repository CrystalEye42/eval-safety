from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--use_system_prompt', action='store_true', help='Append system prompt to start')
    parser.add_argument('--max_length', type=int, default=1000, help='Maximum context length')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = LlamaForCausalLM.from_pretrained('NousResearch/Llama-2-7b-chat-hf', cache_dir='llm_weights').to(device)
    #model = LlamaForCausalLM.from_pretrained('pruned/unstructured_50', cache_dir='llm_weights').to(device)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir='llm_weights').to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir='llm_weights', use_fast=True)
    
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    
    if args.use_system_prompt:
        orig_prompt = f"<s>[INST] <<SYS>> \n{system_prompt} \n<</SYS>> \n\n"
    else:
        orig_prompt = "<s>[INST] "
    old_prompt = orig_prompt

    while True:
        prompt = input("User: ")
        print("-----")
        if prompt == "q":
            break
        if prompt == "r":
            old_prompt = orig_prompt
            print("Context reset!")
            print("-----")
            continue
            
        prompt = old_prompt + prompt + " [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
        # Generate
        generate_ids = model.generate(inputs.input_ids, max_length=args.max_length)
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]#.split("[/INST]")[-1]
        print(response)
        print("-----")
        old_prompt = prompt + "  " + response + " </s><s>[INST] "

if __name__=="__main__":
    main()
