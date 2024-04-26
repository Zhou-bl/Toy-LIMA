import json
# read data from expanded_1500.json
file_path = 'expanded_1500.jsonl'

# Open the file and load the data

system_prompt = open('sys_prompt.txt', 'r').read()

data = []
with open(file_path, 'r') as file:
    for line in file:
        try:
            obj = json.loads(line)
            data.append(obj)
        except json.JSONDecodeError as e:
            print(f"Failed to decode line: {line}")
            print(f"Error: {e}")

for i in range(0, len(data)):
    data[i]["system"] = system_prompt

with open(file_path, 'w') as file:
    for entry in data:
        json.dump(entry, file)
        file.write('\n')