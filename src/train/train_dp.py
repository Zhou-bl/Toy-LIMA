import huggingface_hub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, AdamW, get_scheduler, Qwen2Tokenizer
from datasets import load_dataset
import json
from torch.utils.data import DataLoader
import argparse
import tqdm
import os
from torch.utils.data import Dataset
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the model check point')
parser.add_argument('--system_prompt', type=str, default='data/sys_prompt.txt')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument("--local_rank", default=-1, type=int)

args = parser.parse_args()
local_rank = args.local_rank

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
IGNORE_INDEX = -100  

class mydataLoader(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    #print("device:", device)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(local_rank)
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
        user.append(d['instruction'])
        if d['instances'][0]['input'] != '':
            user[-1] = user[-1] + ' ' + d['instances'][0]['input']
        assistant.append(d['instances'][0]['output'])
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
    # res = []
    # cur_index = 0
    # while cur_index < len(data):
    #     input_ids, labels = [], []
    #     l = cur_index
    #     r = min(cur_index + args.batch_size, len(data))
    #     for i in range(l, r):
    #         input_ids.append(data[i]["input_ids"])
    #         labels.append(data[i]["labels"])
    #     cur_index = r
    #     input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    #     labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    #     attention_mask = (input_ids != tokenizer.pad_token_id).long()
    #     assert input_ids.shape == labels.shape
    #     assert input_ids.shape == attention_mask.shape
    #     res.append({"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask})
    # res = mydataLoader(res)
    res = mydataLoader(data)
    return res

def infer_test(model, tokenizer):
    prompt = "Helle, can you introduce yourself?"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    print(input_ids)
    output = model.generate(input_ids, max_length=100, temperature=0.9)
    print(output)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    return None

def train(model, data):
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(data) * args.epoch
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    print("[INFO] Training steps:", num_training_steps)
    model.train()
    loss_step, loss_epoch = [], []
    for epoch in range(args.epoch):
        print("[INFO] Epoch {} begin".format(epoch))
        epoch_loss = 0
        for batch in tqdm.tqdm(data):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            print("loss: ", loss)
            loss_step.append(loss.item())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        loss_epoch.append(epoch_loss / len(data))
        print("[INFO] Epoch {} end, avg loss: {}".format(epoch, epoch_loss / len(data)))
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_{}.pt".format(epoch)))
    
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

def setup():
    # args.vis_gpu = args.vis_gpu.split(',')
    # for item in args.vis_gpu:
    #     device_ids.append(int(item))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

def main():
    setup()
    # print("devices ids:", device_ids)
    #infer_test(Model, Tokenizer)
    Model, Tokenizer = load_model(args.model_path)
    data = load_data(args.data_file, Tokenizer)
    sampler = torch.utils.data.distributed.DistributedSampler(data)
    data_loader = DataLoader(data, batch_size=args.batch_size, sampler=sampler)
    Model = torch.nn.parallel.DistributedDataParallel(Model, device_ids=[local_rank], output_device=local_rank)
    train(Model, data_loader)


if __name__ == '__main__':
    main()