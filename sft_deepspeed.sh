export MODEL_NAME="qwen1.8B"
export DATA_DIR="/amax/zbl/toy_LIMA/data"
export DATA_NAME="onwdata_highQ"
export BASE_MODEL="/amax/zbl/toy_LIMA/Qwen/Qwen1.5-1.8B" # JUST AN EXAMPLE

cd LLaMA-Factory
CUDA_VISIBLE_DEVICES="2"
deepspeed --num_gpus 1 src/train_bash.py \
    --deepspeed /amax/zbl/toy_LIMA/LLaMA-Factory/examples/deepspeed/ds_z2_offload_config.json \
    --stage sft \
    --do_train True \
    --model_name_or_path ${BASE_MODEL} \
    --finetuning_type full \
    --template qwen \
    --dataset_dir ${DATA_DIR} \
    --dataset ${DATA_NAME} \
    --cutoff_len 512 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --preprocessing_num_workers 8 \
    --max_steps 2000 \
    --save_steps 200 \
    --warmup_steps 100 \
    --output_dir checkpoints/${MODEL_NAME} \
    --bf16 True \
    --plot_loss True \
    --overwrite_output_dir