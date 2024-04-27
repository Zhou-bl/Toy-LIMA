# Toy LIMA

This is the repo for the toy LIMA project(homework 2 for cs296:Large Language Model in SJTU).

## Dataset Generation

see `./data/README.md`

## Evaluation

```
alpaca_eval evaluate --model_outputs='path/to/output.json' --annotators_config='chatgpt' 
```

See the `./final_eval` to see the model's output:


```
final_eval
├── Qwen0.5B_baseline.json
├── Qwen0.5B_improved_sft.json
├── Qwen0.5B_naive_sft.json
├── Qwen1.8B_baseline.json
├── Qwen1.8B_improved_sft.json
└── Qwen1.8B_naive_sft.json
```