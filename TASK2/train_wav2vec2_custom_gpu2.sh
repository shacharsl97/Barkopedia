#!/bin/bash

# Training script for Wav2Vec2 Custom model on GPU 2
echo "=== Starting Wav2Vec2 Custom Model Training on GPU 2 ==="

cd /home/cs/weidena1/Barkopedia/TASK2

python train_multi_model.py \
    --model_type wav2vec2_custom \
    --model_name "facebook/wav2vec2-base" \
    --pooling_mode mean \
    --device cuda:2 \
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
    --save_dir "./wav2vec2_custom_training_results" \
    --cache_dir "./barkopedia_gender_cache"

echo "=== Wav2Vec2 Custom Model Training Completed ==="
