import json
import openai
from openai import ChatCompletion
import tokenize
import io

openai.api_base = "https://lonlie.plus7.plus/v1"

def extract_data_from_jsonl(file_path):
    res = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            name = data.get('name', '')
            instruction = data.get('instruction', '')
            res += '{\"' + name + '\", \"' + instruction + '\"}, '
    return res

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

def count_tokens(source_code):
    try:
        tokens = tokenize.tokenize(io.BytesIO(source_code.encode('utf-8')).readline)
        return sum(1 for token in tokens if token.type != tokenize.ENDMARKER)
    except tokenize.TokenError as e:
        print("Tokenization error:", e)
        return None

def convert(input_string):
    escaped_string = input_string.replace("\n", "\\n").replace("\t", "\\t")
    return escaped_string


def main():
    jsonl_file_path = './seed_tasks.jsonl'
    text = extract_data_from_jsonl(jsonl_file_path)
    
    prompt = "Following are some examples of {name, instruction} pairs. Please generate more pairs like these pairs. Please generate more pairs!!! Keep format the same. Generate more pairs! generate more pairs! generate more pairs as long as you can! Be more creative! Following: " + text

    print("Querying:", prompt)
    result = query_openai(prompt)
    print("Result:", result)

if __name__ == '__main__':
    main()

