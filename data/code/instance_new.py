import openai
from openai import ChatCompletion
import json

openai.api_base = "https://lonlie.plus7.plus/v1"

def append_to_jsonl(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(data + '\n')

specific_data = {
    "name": "breakfast_suggestion",
    "instruction": "Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?",
    "instances": [
        {"input": "", "output": "Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams whey protein powder, 1/2 medium banana, 1tbsp flaxseed oil and 1/2 cup water, totaling about 550 calories. The 4 strips of bacon contains about 200 calories."},
        {"input": "", "output": "Yes, you can have a Greek yogurt parfait with granola and mixed berries. The Greek yogurt parfait may contain 1 cup Greek yogurt, 1/2 cup granola, and 1/2 cup mixed berries, totaling about 800 calories."},
        {"input": "", "output": "Yes, you can have a turkey and avocado wrap with a side of cottage cheese. The turkey and avocado wrap may contain 2 slices of turkey, 1/4 avocado, lettuce, and tomato in a whole wheat wrap, and the cottage cheese contains about 150 calories, totaling about 750 calories."},
        {"input": "", "output": "Yes, you can have a protein pancake stack with peanut butter and a glass of almond milk. The protein pancake stack may contain 3 protein pancakes, 1tbsp peanut butter, and a glass of almond milk, totaling about 900 calories."},
        {"input": "", "output": "Yes, you can have a breakfast burrito with black beans, salsa, and guacamole. The breakfast burrito may contain scrambled tofu, black beans, salsa, and guacamole in a whole wheat tortilla, totaling about 700 calories."}
    ]
}

def extract_data_from_jsonl(read_path, write_path):

    with open(read_path, 'r', encoding='utf-8') as file:
        cnt = 0
        for line in file:
            data = json.loads(line)
            query_dict = {
                "name": data.get('name', ''),
                "instruction": data.get('instruction', ''),
            }
            prompt = {
                "prompt": f"Following query_data is a json line with name, instruction. Please generate 5 instances according to given example_data, and return me with the same format of example_data in a jsonl format. You need to reference and return in the format of this: " + json.dumps(specific_data) + ", but the name, instruction and instances should be according to example_data! Use content of example_data and format of upper data!",
                "use_content_data": query_dict
            }
            query = json.dumps(prompt)
            cnt += 1
            res = query_openai(query)
            print(f"Answer {cnt}: {res}")
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
    extract_data_from_jsonl('./output.jsonl', './instance.jsonl')
    

if __name__ == '__main__':
    main()

