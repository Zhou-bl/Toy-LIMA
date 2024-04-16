import huggingface_hub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, AdamW, get_scheduler, Qwen2Tokenizer
from transformers.integrations import HfDeepSpeedConfig
from datasets import load_dataset
import json
from torch.utils.data import DataLoader
import argparse
import tqdm
import deepspeed
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the model check point')
parser.add_argument('--system_prompt', type=str, default='data/sys_prompt.txt')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument("--local_rank", type=int, default=0)
parser = deepspeed.add_config_arguments(parser)

args = parser.parse_args()

IGNORE_INDEX = -100

def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
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

def infer_test(model, tokenizer):
    prompt = "Helle, can you introduce yourself?"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    print(input_ids)
    output = model.generate(input_ids, max_length=100, temperature=0.9)
    print(output)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    return None

def train(engine, data):

    loss_step, loss_epoch = [], []
    for epoch in range(args.epoch):
        print("[INFO] Epoch {} begin".format(epoch))
        epoch_loss = 0
        for batch in tqdm.tqdm(data):
            batch = {k: v for k, v in batch.items()}
            loss = engine(**batch)
            loss_step.append(loss.item())
            epoch_loss += loss.item()
            engine.backward(loss)
            engine.step()
        loss_epoch.append(epoch_loss / len(data))
        print("[INFO] Epoch {} end, avg loss: {}".format(epoch, epoch_loss / len(data)))
        engine.save_checkpoint(f"{args.save_dir}/epoch_{epoch}.pt")
    
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
    ds_config = json.load(open(args.deepspeed_config, 'r'))
    dschf = HfDeepSpeedConfig(ds_config)
    Model, Tokenizer = load_model(args.model_path)
    Engine = deepspeed.initialize(
        model = Model,
        config_params=ds_config
    )
    #infer_test(Model, Tokenizer)
    data = load_data(args.data_file, Tokenizer)
    train(Engine, data)

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    main()