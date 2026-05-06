#!/bin/bash
# Vanilla 2x-parameter baseline (Llama-410m with the layer count doubled, 24 -> 48 layers).
# Roughly 820M parameters; serves as the "twice the size" reference for the 410m comparison.

set -e

OUTPUT_DIR=${OUTPUT_DIR:-saves/smallpile/vanilla_llama_410m_x2depth}
MODEL_CFG=${MODEL_CFG:-llama_config/410m_x2depth}
TOKENIZED=${TOKENIZED:-data/tokenized_data/smallpile}

python -m llamafactory.launcher \
    --model_name_or_path $MODEL_CFG \
    --stage pt \
    --do_train --do_eval \
    --finetuning_type full \
    --train_from_scratch \
    --tokenized_path $TOKENIZED \
    --dataset smallpile \
    --eval_dataset testpile \
    --template default \
    --max_steps 50000 \
    --save_steps 5000 \
    --save_total_limit 3 \
    --eval_steps 5000 \
    --evaluation_strategy steps \
    --logging_steps 1 \
    --plot_loss \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 3e-4 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
    --warmup_ratio 0.02 \
    --adam_beta1 0.9 --adam_beta2 0.95 \
    --weight_decay 0.01 \
    --bf16 \
    --flash_attn fa2 \
    --disable_gradient_checkpointing true \
    --dataloader_num_workers 16 \
    --ddp_timeout 180000000 \
    --deepspeed examples/deepspeed/ds_z0_config.json \
    --report_to wandb \
    --output_dir $OUTPUT_DIR
