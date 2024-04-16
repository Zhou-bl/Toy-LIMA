gpu_vis=""
MASTER_PORT=22345

deepspeed \
--include localhost:$gpu_vis \
--master_port $MASTER_PORT \
src/train/sft.py \
--model_path Qwen/Qwen1.5-1.8B \
--data_file data/seed_tasks.jsonl \
--save_dir save \
--batch_size 8 \
--deepspeed \
--deepspeed_config deepspeed/ds_config.json