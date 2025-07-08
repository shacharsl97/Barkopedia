#!/bin/bash

# Test script for multi-model training
echo "=== Testing Multi-Model Training Script ==="

cd /home/cs/weidena1/Barkopedia/TASK2

# Test AST model
echo "Testing AST model..."
python train_multi_model.py \
    --model_type ast \
    --model_name "MIT/ast-finetuned-audioset-10-10-0.4593" \
    --device cuda:0 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --eval_steps 10 \
    --save_steps 20 \
    --log_steps 5 \
    --enable_segmentation \
    --apply_cleaning \
    --create_plots \
    --debug \
    --save_dir "./test_ast_results"

echo "AST model test completed!"

# Test Wav2Vec2 Custom model
echo "Testing Wav2Vec2 Custom model..."
python train_multi_model.py \
    --model_type wav2vec2_custom \
    --model_name "facebook/wav2vec2-base" \
    --pooling_mode mean \
    --device cuda:0 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --eval_steps 10 \
    --save_steps 20 \
    --log_steps 5 \
    --enable_segmentation \
    --apply_cleaning \
    --create_plots \
    --debug \
    --save_dir "./test_wav2vec2_custom_results"

echo "Wav2Vec2 Custom model test completed!"

# Test Wav2Vec2 Pretrained model
echo "Testing Wav2Vec2 Pretrained model..."
python train_multi_model.py \
    --model_type wav2vec2_pretrained \
    --model_name "facebook/wav2vec2-base" \
    --device cuda:0 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --eval_steps 10 \
    --save_steps 20 \
    --log_steps 5 \
    --enable_segmentation \
    --apply_cleaning \
    --create_plots \
    --debug \
    --save_dir "./test_wav2vec2_pretrained_results"

echo "Wav2Vec2 Pretrained model test completed!"

echo "=== All tests completed! ==="
