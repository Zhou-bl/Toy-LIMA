import json

def validate_and_split_jsonl(input_path, valid_output_path, invalid_output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    with open(valid_output_path, 'a', encoding='utf-8') as valid_file, \
         open(invalid_output_path, 'w', encoding='utf-8') as invalid_file:
        for line in lines:
            try:
                json_object = json.loads(line)
                valid_file.write(line)
            except json.JSONDecodeError:
                invalid_file.write(line)

validate_and_split_jsonl('data_generated.jsonl', 'data_checked.jsonl', 'data_failed.jsonl')
