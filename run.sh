export MODEL_PATH=$1
export SAVE_PATH=$3
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export WANDB_DISABLED=true
wandb offline

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=2 --use_env train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $2 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size  \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --gradient_checkpointing True
