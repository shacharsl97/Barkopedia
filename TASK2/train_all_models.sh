#!/bin/bash

# Master script to run all three model training jobs in parallel
echo "=== Starting Multi-Model Training Pipeline ==="
echo "Training AST on GPU 0, Wav2Vec2 Custom on GPU 2, and Wav2Vec2 Pretrained on GPU 1"

cd /home/cs/weidena1/Barkopedia/TASK2

# Make scripts executable
chmod +x train_ast_gpu0.sh
chmod +x train_wav2vec2_custom_gpu2.sh
chmod +x train_wav2vec2_pretrained_gpu1.sh

# Start all training jobs in parallel
echo "Starting AST training on GPU 0..."
nohup ./train_ast_gpu0.sh > ast_training.log 2>&1 &
AST_PID=$!

echo "Starting Wav2Vec2 Custom training on GPU 2..."
nohup ./train_wav2vec2_custom_gpu2.sh > wav2vec2_custom_training.log 2>&1 &
WAV2VEC2_CUSTOM_PID=$!

echo "Starting Wav2Vec2 Pretrained training on GPU 1..."
nohup ./train_wav2vec2_pretrained_gpu1.sh > wav2vec2_pretrained_training.log 2>&1 &
WAV2VEC2_PRETRAINED_PID=$!

echo "All training jobs started!"
echo "AST PID: $AST_PID"
echo "Wav2Vec2 Custom PID: $WAV2VEC2_CUSTOM_PID"
echo "Wav2Vec2 Pretrained PID: $WAV2VEC2_PRETRAINED_PID"

echo ""
echo "You can monitor progress with:"
echo "  tail -f ast_training.log"
echo "  tail -f wav2vec2_custom_training.log"
echo "  tail -f wav2vec2_pretrained_training.log"
echo ""
echo "Or check GPU usage with:"
echo "  nvidia-smi"
echo ""
echo "To check if jobs are still running:"
echo "  ps aux | grep train_multi_model.py"

# Wait for all jobs to complete
wait $AST_PID
echo "AST training completed!"

wait $WAV2VEC2_CUSTOM_PID
echo "Wav2Vec2 Custom training completed!"

wait $WAV2VEC2_PRETRAINED_PID
echo "Wav2Vec2 Pretrained training completed!"

echo "=== All training jobs completed! ==="
echo "Check the respective result directories for trained models and plots."
