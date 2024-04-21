import json

file_name = 'baseline.json'

with open(file_name, 'r', encoding='utf-8') as file:
    data = json.load(file)

for item in data:
    instr = item['instruction'] + ' '
    if item['output'].startswith(instr):
        item['output'] = item['output'].replace(instr, '', 1)

with open(file_name, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)