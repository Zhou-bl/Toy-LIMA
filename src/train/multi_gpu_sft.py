import huggingface_hub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, AdamW, get_scheduler, Qwen2Tokenizer
from datasets import load_dataset
import json
from torch.utils.data import DataLoader
import argparse
import tqdm
import deepspeed
import datasets
from deepspeed import get_accelerator
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the model check point')
parser.add_argument('--system_prompt', type=str, default='data/sys_prompt.txt')
parser.add_argument('--eval', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument("--local_rank", type=int, default=0)
parser = deepspeed.add_config_arguments(parser)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
IGNORE_INDEX = -100  

def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    #print("device:", device)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    if tokenizer.pad_token is None:
        print("[INFO] Add pad token to tokenizer.")
        tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def load_data(path, tokenizer):
    user = []
    assistant = []
    file = open(path, 'r')
    data = []
    for line in file:
        data.append(json.loads(line))
    for d in data:
        for i in range(len(d['instances'])):
            user.append(d['instruction'])
            if d['instances'][i]['input'] != '':
                user[-1] = user[-1] + ' ' + d['instances'][i]['input']
            assistant.append(d['instances'][i]['output'])
        # user.append(d['instruction'])
        # if d['instances'][0]['input'] != '':
        #     user[-1] = user[-1] + ' ' + d['instances'][0]['input']
        # assistant.append(d['instances'][0]['output'])
    assert len(user) == len(assistant)
    # print("User:", user[0])
    # print("Assistant:", assistant[0])
    system_prompt = open(args.system_prompt, 'r').read()
    # print("system_prompt:", system_prompt)
    for i in range(len(user)):
        user[i] = system_prompt + '\n' + "User: " + user[i]
        assistant[i] = assistant[i] + tokenizer.eos_token
    assit_head = "Assistant: "
    # Build input ids and label
    data = []
    for i in range(len(user)):
        tokenized_ids = []
        labels = []
        tokenized = tokenizer(user[i], add_special_tokens=False)
        tokenized_ids += tokenized["input_ids"]
        labels += ([IGNORE_INDEX] *  len(tokenized["input_ids"]))
        tokenized = tokenizer(assit_head, add_special_tokens=False)
        tokenized_ids += tokenized["input_ids"]
        labels += ([IGNORE_INDEX] *  len(tokenized["input_ids"]))
        tokenized = tokenizer(assistant[i], add_special_tokens=False)
        tokenized_ids += tokenized["input_ids"]
        labels += tokenized["input_ids"]
        data.append({"input_ids": torch.tensor(tokenized_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)})

    # print("len: ", len(data))
    # print(data[0])
    # Batch the data
    '''
    return a list of dictionary, each dictionary contains: input_ids, labels, attention mask
    '''
    res = []
    cur_index = 0
    while cur_index < len(data):
        input_ids, labels = [], []
        l = cur_index
        r = min(cur_index + args.batch_size, len(data))
        for i in range(l, r):
            input_ids.append(data[i]["input_ids"])
            labels.append(data[i]["labels"])
        cur_index = r
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        assert input_ids.shape == labels.shape
        assert input_ids.shape == attention_mask.shape
        res.append({"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask})
    return res

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

def get_output(model, tokenizer, input):
    # input is a list of strings
    system_prompt = open(args.system_prompt, 'r').read()
    input = system_prompt + '\n' + "User: " + input + "Assistant: "
    input_ids = tokenizer(input, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    output_ids = model.generate(input_ids, 
                                max_length=1024, 
                                do_sample=True, 
                                min_new_tokens=2, 
                                max_new_tokens=256)
    # print("input_ids:", input_ids)
    # print("output_ids:", output_ids)
    output_ids = output_ids[:,input_ids.shape[-1]:]
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return output

def eval(model, tokenizer, file, epoch):
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
        with open(file+"/"+str(epoch)+".json", "w", encoding="utf-8") as f:
            f.write(json.dumps(json_list, ensure_ascii=False, indent=4))

def train(model, tokenizer, data):
    num_training_steps = len(data) * args.epoch
    print("[INFO] Training steps:", num_training_steps)
    model.train()
    loss_step, loss_epoch = [], []
    for epoch in range(args.epoch):
        print("[INFO] Epoch {} begin".format(epoch))
        epoch_loss = 0
        for batch in tqdm.tqdm(data):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model.forward(input_ids=batch["input_ids"], labels=batch["labels"], attention_mask=batch["attention_mask"])
            #print("loss:", loss)
            # epoch_loss += loss.items
            loss = output.loss
            loss_step.append(loss.item())
            epoch_loss += loss_step[-1]
            model.backward(loss)
            model.step()
            get_accelerator().empty_cache()
        loss_epoch.append(epoch_loss / len(data))
        print("[INFO] Epoch {} end, avg loss: {}".format(epoch, epoch_loss / len(data)))
        eval(model, tokenizer, args.eval, epoch)
        model.save_pretrained(args.save_dir + '/epoch_{}'.format(epoch))
    
    plt.figure("Step loss")
    plt.plot(loss_step)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("fig/step_loss.png")
    plt.figure("Epoch loss")
    plt.plot(loss_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("fig/epoch_loss.png")
    return None


def main():
    #infer_test(Model, Tokenizer)
    deepspeed.init_distributed(dist_backend="nccl")
    Model, Tokenizer = load_model(args.model_path)
    data = load_data(args.data_file, Tokenizer)
    deepspeed_config = "deepspeed/ds_config.json"
    model_engine, _, trainloader, _ = deepspeed.initialize(
        args=args,
        model=Model,
        model_parameters=Model.parameters(),
        training_data=data,
        config=deepspeed_config
    )
    
    print("local rank: ", model_engine.local_rank)
    train(model_engine, Tokenizer, data)

if __name__ == '__main__':
    main()