# Toy LIMA

This is the repo for the toy LIMA project(homework 2 for cs296:Large Language Model in SJTU).

## Requirements

```bash
conda create -n toy-lima python=3.10
conda activate toy-lima
pip install -r requirements.txt
pip install deepspeed #[optional]
```

## Construct the dataset

In this project, we construct the sft dataset by three ways: two simple, naive construct methods and one more complex method base on the naive methods.

The details of dataconstruct methods can be seen in our `minipaper`.

`data/naive_tasks.jsonl` and `data/naive_expand.json` are the data generated by our two naive methods. While `data/expanded_1000.jsonl` and `data/expanded_1500.jsonl` are the data generated by our complex method with different data example quantity.

## Supervised Fine-tuning

We use `LLaMa-Factory` to fine-tune the model. The hyperparameter details of the fine-tuning process can be seen in our `minipaper`.

You can run the fine-tuning process by running the following command:

```bash
bash sft.sh
```

If your GPU memory is not enough, you can run the fine-tuning process with deepspeed ZeRo-2 with CPU offload by running the following command:

```bash
bash sft_deepspeed.py
```

## Evaluation

We use `AlpacaEval 2.0` to eval our model. To get the output from a specific checkpoint, run the following command(use the 0.5B 200 step checkpoint as an example):

```bash
python infer.py \
--model_path /amax/zbl/toy_LIMA/LLaMA-Factory/checkpoints/qwen0.5B/checkpoint-200 \
--tokenizer_path /amax/zbl/toy_LIMA/LLaMA-Factory/checkpoints/qwen0.5B \
--file 2024-4-26/Qwen0.5B_new/checkpoint-200.json
```

Then you can use `AlpacaEval 2.0` to evaluate the quality of model's output.

For the overall result, you can refer to our minipaper.