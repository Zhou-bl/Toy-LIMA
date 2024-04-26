from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import tqdm
import datasets
import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--tokenizer_path', type=str, required=True)
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--system_prompt', type=str, default="data/sys_prompt.txt")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

def get_output(model, tokenizer, input):
    # input is a list of strings
    system_prompt = open(args.system_prompt, 'r').read()
    input = system_prompt + '\n' + "User: " + input + "Assistant: "
    input_ids = tokenizer(input, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    output_ids = model.generate(input_ids, 
                                max_length=1024, 
                                do_sample=True, 
                                min_new_tokens=2,
                                max_new_tokens=512)
    # print("input_ids:", input_ids)
    # print("output_ids:", output_ids)
    output_ids = output_ids[:,input_ids.shape[-1]:]
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return output

def eval(model, tokenizer, file):
    json_list = []
    for example in tqdm.tqdm(eval_set):
        #print("example:", example)
        #generate here is a placeholder for your models generations
        instruction = example["instruction"]
        try:
            output = get_output(model, tokenizer, instruction)
        except:
            print("[ERROR]!!!")
            output = ""
        json_list.append({"instruction": instruction, "output": output})
        with open(file, "w", encoding="utf-8") as f:
            f.write(json.dumps(json_list, ensure_ascii=False, indent=4))

def main():
    eval(model, tokenizer, args.file)

if __name__ == "__main__":
    main()