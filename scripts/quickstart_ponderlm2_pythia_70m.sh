#!/bin/bash
# Quickstart: train a tiny PonderLM-2 (Pythia-70m) on minipile.
# Targets an 8-GPU node; runs in well under an hour on 8×RTX4090.

set -e

OUTPUT_DIR=${OUTPUT_DIR:-saves/minipile/ponderlm2_pythia_70m}

FORCE_TORCHRUN=1 llamafactory-cli train \
    --model_name_or_path EleutherAI/pythia-70m \
    --stage pt --do_train --finetuning_type full \
    --train_from_scratch true \
    --dataset minipile \
    --template default \
    --cutoff_len 2048 \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1.0e-3 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
    --warmup_ratio 0.01 \
    --adam_beta1 0.9 --adam_beta2 0.95 \
    --weight_decay 0.01 \
    --bf16 \
    --flash_attn fa2 \
    --disable_gradient_checkpointing true \
    --preprocessing_num_workers 16 \
    --logging_steps 1 --save_steps 2000 --plot_loss \
    --report_to wandb \
    --ddp_timeout 180000000 \
    --deepspeed examples/deepspeed/ds_z0_config.json \
    --patch_method ours \
    --recurrent_model true \
    --scale_embeds true \
    --interpolation false \
    --num_latent_thoughts 1 \
    --num_jacobi_iterations 3 \
    --random_jacobi_iterations true \
    --output_dir $OUTPUT_DIR
