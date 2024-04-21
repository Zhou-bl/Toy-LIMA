import json
import openai
from openai import OpenAI
import tokenize
import io

client = OpenAI(
    api_key="sk-baggqvfu5XAaZwtL779383E07eDe4b1589Bd99F6E88f8e8e",
    base_url="https://lonlie.plus7.plus/v1"
)

def extract_data_from_jsonl(file_path):
    res = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            name = data.get('name', '')
            instruction = data.get('instruction', '')
            res += '{\"' + name + '\", \"' + instruction + '\"}, '
    return res

def query_openai(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
      	model=model,
      	messages=messages,
      	temperature=0.7,
    )
    return response.choices[0].message.content

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
    
#    print(convert(text))
#    print(count_tokens(convert(text)))

    prompt = "Following are 174 examples of {name, instruction} pairs. Please generate more pairs like these pairs. Please generate 326 pairs. Keep format the same. Here are example pairs: " + text

    print("Querying:", prompt)
    result = query_openai(prompt=prompt)
    print("Result:", result)

# 执行主函数
if __name__ == '__main__':
    main()

