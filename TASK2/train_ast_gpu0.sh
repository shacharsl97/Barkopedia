#!/bin/bash

# Training script for AST model on GPU 0
echo "=== Starting AST Model Training on GPU 0 ==="

cd /home/cs/weidena1/Barkopedia/TASK2

python train_multi_model.py \
    --model_type ast \
    --model_name "MIT/ast-finetuned-audioset-10-10-0.4593" \
    --device cuda:0 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --weight_decay 0.01 \
    --eval_steps 400 \
    --save_steps 400 \
    --log_steps 50 \
    --enable_segmentation \
    --segment_duration 2.0 \
    --segment_overlap 0.1 \
    --energy_threshold 0.01 \
    --apply_cleaning \
    --create_plots \
    --save_dir "./ast_training_results" \
    --cache_dir "./barkopedia_gender_cache"

echo "=== AST Model Training Completed ==="
