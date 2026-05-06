#!/bin/bash
# Loop baseline (3 extra layer-stack passes) on Llama-410m, smaller-compute companion to Table 3.

set -e

OUTPUT_DIR=${OUTPUT_DIR:-saves/smallpile/loop_llama_410m}
MODEL_CFG=${MODEL_CFG:-llama_config/410m}
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
    --max_position_embeddings 4096 \
    --max_steps 50000 \
    --save_steps 5000 \
    --save_total_limit 3 \
    --eval_steps 5000 \
    --evaluation_strategy steps \
    --logging_steps 1 \
    --plot_loss \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
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
    --patch_method loop \
    --recurrent_model true \
    --scale_embeds true \
    --interpolation false \
    --num_latent_thoughts 3 \
    --random_jacobi_iterations true \
    --num_jacobi_iterations 3 \
    --output_dir $OUTPUT_DIR
