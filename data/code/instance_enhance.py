import json
import time
import openai
from openai import ChatCompletion
import tokenize
import io

# openai.api_base = "https://lonlie.plus7.plus/v1"
openai.api_base = "https://www.jcapikey.com/v1"

def append_to_jsonl(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(data + '\n')

eg_name = "breakfast_suggestion"
eg_instruction = "Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?"
eg_input = ""
eg_output = "Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams whey protein powder, 1/2 medium banana, 1tbsp flaxseed oil and 1/2 cup water, totaling about 550 calories. The 4 strips of bacon contains about 200 calories."
eg_expand = "For a delightful morning indulgence sans eggs but brimming with nourishing protein and energy, consider a sumptuous Greek yogurt parfait kissed by the sun-ripened sweetness of vibrant mixed berries and the satisfying crunch of artisanal granola. Picture a lavish blend of velvety Greek yogurt, generously layered with crunchy clusters of golden granola, each bite releasing a symphony of wholesome flavors and textures. Embark on a gastronomic journey where creamy yogurt meets the luscious juiciness of assorted berries, creating a harmonious dance of tartness and sweetness on your palate. This tantalizing parfait, meticulously crafted with 1 cup of velvety Greek yogurt, a lavish sprinkling of 1/2 cup of wholesome granola, and a generous scattering of 1/2 cup of succulent mixed berries, promises a tantalizing sensory experience that will awaken your taste buds and energize your day. Indulge guilt-free in this culinary masterpiece, meticulously curated to provide a perfect balance of nutrition, flavor, and satisfaction, as you savor every spoonful of this breakfast bliss!"


def extract_data_from_jsonl(read_path, write_path):

    with open(read_path, 'r', encoding='utf-8') as file:
        cnt = 0
        for line in file:
            data = json.loads(line)

            name_str = data.get('name', '')
            instruction_str = data.get('instruction', '')

            # query_dict = {
            #     "name": data.get('name', ''),
            #     "instruction": data.get('instruction', '')
            #     # "instances": data.get('instances', [])
            # }
            cnt += 1
            if cnt <= 162: # 229 done
                continue

            try: 
                for instance in data['instances']:
                    # print("Input:", instance['input'])
                    print("#Original:", instance['output'])

                    prompt = f"Given is a question-answer task with task name, instruction and input-output instance. The output text is too short, insipid, dull, boring. I want it to be more Innovative, Imaginative, Comprehensive, Thorough, Exhaustive, Exact, creative, detailed, interesting, engaging, fascinating, and specific. Add at least 10 words, more words are better. Generate more creative words for the output part according to the task name, instruction and input. Here is a example: Data give to you: Task name: '{eg_name}', Instruction: '{eg_instruction}', Input: '{eg_input}, output: '{eg_output}'. Your output should be: {eg_expand}. Following is the task: Task name: '{name_str}', Instruction: '{instruction_str}', Input: '{instance['input']}, output: '{instance['output']}'. You need to directly generate the output part, DO NOT SAY any other words! print output text directly! "
                    
                    query = json.dumps(prompt)
                    res = query_openai(query)
                    print("#Expanded:", res)

                    record = {
                        "name": name_str,
                        "instruction": instruction_str,
                        "input": instance['input'],
                        "output":res 
                    }

                    json_line = json.dumps(record)
                    append_to_jsonl(write_path, json_line)
                print(f"$$$$$$ Answer {cnt} done. $$$$$$$")
            except Exception as e:
                print(f"Error: {e}")
                continue


def query_openai(prompt):
    retries = 10 
    wait_time = 1 

    for i in range(retries):
        try:
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
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time *= 2
            if (wait_time > 32):
                wait_time = 1
    
    print("ERROR")
    exit()


def main():
    extract_data_from_jsonl('./data_generated.jsonl', './data_enhanced.jsonl')
    

if __name__ == '__main__':
    main()

