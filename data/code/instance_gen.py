import json
import openai
from openai import ChatCompletion
import tokenize
import io

openai.api_base = ""

def append_to_jsonl(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(data + '\n')


def extract_data_from_jsonl(read_path, write_path):

    with open(read_path, 'r', encoding='utf-8') as file:
        cnt = 0
        for line in file:
            data = json.loads(line)
            query_dict = {
                "name": data.get('name', ''),
                "instruction": data.get('instruction', ''),
                "instances": data.get('instances', [])
            }
            prompt = {
                "prompt": f"Following is a json line with name, instruction, and instances. There are some instances give to you. Please generate 2 more instances according to given information, add them to the jsonl I give you, and return me with the same format in a line.",
                "data": query_dict
            }
            query = json.dumps(prompt)
            cnt += 1
            res = query_openai(query)
            print(f"Answer {cnt} done.")
            append_to_jsonl(write_path, res)


            

def query_openai(prompt):
    client = ChatCompletion()
    response = client.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.8,
        max_tokens=4096
    )
    return response['choices'][0]['message']['content'].strip()

def main():
    extract_data_from_jsonl('./seed_tasks.jsonl', './tasks.jsonl')
    

if __name__ == '__main__':
    main()

