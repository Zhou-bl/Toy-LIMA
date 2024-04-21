gpu_vis="5"
MASTER_PORT=22345
export PATH=/usr/local/cuda/bin:$PATH \
export CPATH=/usr/local/cuda/include:$CPATH \
#export DEFAULT_TORCH_EXTENSION_PATH="/amax/zbl/.conda/envs/python3.10_env/lib/python3.10/site-packages/deepspeed/ops/op_builder/builder.py"
deepspeed \
--include localhost:$gpu_vis \
--master_port $MASTER_PORT \
src/train/multi_gpu_sft.py \
--model_path /amax/zbl/toy_LIMA/model/Qwen1.5-1.8B \
--data_file data/tasks.jsonl \
--eval result \
--save_dir save \
--batch_size 2 \