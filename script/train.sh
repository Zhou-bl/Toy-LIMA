export CUDA_VISIBLE_DEVICES="0" \
gpu_vis="0"
MASTER_PORT=22346
export PATH=/usr/local/cuda/bin:$PATH \
export CPATH=/usr/local/cuda/include:$CPATH \
#export DEFAULT_TORCH_EXTENSION_PATH="/amax/zbl/.conda/envs/python3.10_env/lib/python3.10/site-packages/deepspeed/ops/op_builder/builder.py"
deepspeed \
--include localhost:$gpu_vis \
--master_port $MASTER_PORT \
src/train/multi_gpu_sft.py \
--model_path Qwen/Qwen1.5-0.5B \
--data_file data/tasks.jsonl \
--eval result_0.5 \
--save_dir save_new \
--epoch 10 \
--batch_size 2 \