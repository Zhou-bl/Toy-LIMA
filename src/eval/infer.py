import huggingface_hub
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import json
import argparse
import tqdm
import datasets
#import deepspeed
import os

def get_output(model, tokenizer, input):
    # input is a list of strings
    input_ids = tokenizer(input, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    output_ids = model.generate(input_ids, max_length=1024, do_sample=True)
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return output

def load_model(model_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    #print("device:", device)
    if tokenizer.pad_token is None:
        print("[INFO] Add pad token to tokenizer.")
        tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.eos_token_id

    with open(model_path + "/config.json", "r") as f:
        config_set = json.load(f)
    print(config_set._name_or_path)
    config = AutoModelForCausalLM.from_config(config_set)
    model = AutoModelForCausalLM(config)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(model_path+"/model.safetensors", map_location=device))
    return model, tokenizer

def batch_data(data, batch_size):
    batched_data = []
    for i in range(0, len(data), batch_size):
        batched_data.append(data[i:i+batch_size])
    return batched_data

def main():
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    #print("raw_eval_set:", raw_eval_set)
    
    json_list = []
    # 截取dataset的前15个
    # eval_set = batch_data(eval_set, args.batch_size)
    # print("batch_set", eval_set[0])
    # print("batch_set", eval_set[1])
    # print("batch_set", eval_set[2])
    model, tokenizer = load_model(args.model_path, args.tokenizer_path)
    for example in tqdm.tqdm(eval_set):
        #print("example:", example)
        #generate here is a placeholder for your models generations
        instruction = example["instruction"]
        try:
            output = get_output(model, tokenizer, instruction)
        except:
            print("[Warning] get_output failed, instruction:", instruction)
            output = ""
        json_list.append({"instruction": instruction, "output": output})
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(json_list, ensure_ascii=False, indent=4))
        
argparser = argparse.ArgumentParser()
argparser.add_argument("--tokenizer_path", type=str, required=True)
argparser.add_argument("--model_path", type=str, required=True)
argparser.add_argument("--output_path", type=str, required=True)
#argparser.add_argument("--model_name", type=str, required=True)
args = argparser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    main()