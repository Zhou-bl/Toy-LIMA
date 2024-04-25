CUDA_VISIBLE_DEVICES="1,2,3" \
python -m torch.distributed.launch --nproc_per_node 1 src/train/train_dp.py \
--model_path /amax/zbl/toy_LIMA/Qwen/Qwen1.5-0.5B \
--data_file data/seed_tasks.jsonl \
--save_dir save \
--batch_size 8 \