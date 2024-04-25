python src/train/sft.py \
--model_path /amax/zbl/toy_LIMA/Qwen/Qwen1.5-0.5B \
--data_file data/expanded_seed_tasks.jsonl \
--save_dir save \
--eval result_0.5 \
--batch_size 16 \