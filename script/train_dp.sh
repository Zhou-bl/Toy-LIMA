CUDA_VISIBLE_DEVICES="1" \
python -m torch.distributed.launch --nproc_per_node 1 src/train/train_dp.py \
--model_path /amax/zbl/toy_LIMA/model/Qwen1.5-1.8B \
--data_file data/seed_tasks.jsonl \
--save_dir save \
--batch_size 1 \