This readme file contains some details about the data generation process.  The final employed expanded data is `expanded_1500.jsonl`.

According to the report paper, we provided the data generating python code in the `code/` folder. Following are descriptions about them: 

+ `name_gen.py`: Generating new tasks according to the provided 174 task name and instruction.

+ `instance_gen.py`: For the given-instance tasks, generate more instances by one-shot method.

+ `instance_new.py`: For the new generated tasks, generate more instances by onse-shot method.

+ `instance_enhance.py`: Enrich and polish the output text of each instance.

+ `check.py`: The format of generated text may not meet the json requirements, which can be checked by this file.

