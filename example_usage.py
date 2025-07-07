#!/usr/bin/env python3
"""
Example usage of the Barkopedia AST Classification Model.

This script demonstrates how to:
1. Load the AST classification model
2. Make predictions on audio data
3. Use the forward method for raw logits
"""

import numpy as np
import torch
import sys
import os

# Add the models directory to the path
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
sys.path.insert(0, models_dir)

from models.ast_classification_model import ASTClassificationModel

def main():
    print("=== Barkopedia AST Classification Model Example ===\n")
    
    # Initialize the model
    print("1. Initializing AST Classification Model...")
    model = ASTClassificationModel(num_labels=5)
    
    # Load the pretrained model (or fine-tuned if available)
    print("2. Loading model weights...")
    model.load("MIT/ast-finetuned-audioset-10-10-0.4593")
    
    print(f"   ✓ Model loaded on device: {model.device}")
    print(f"   ✓ Backbone: {model.backbone.backbone_name}")
    print(f"   ✓ Number of output classes: {model.num_labels}")
    
    # Create some dummy audio data (normally you'd load real audio)
    print("\n3. Preparing audio data...")
    sampling_rate = 16000
    duration = 2  # seconds
    dummy_audio = np.random.randn(sampling_rate * duration)
    print(f"   ✓ Audio shape: {dummy_audio.shape}")
    print(f"   ✓ Sampling rate: {sampling_rate} Hz")
    print(f"   ✓ Duration: {duration} seconds")
    
    # Method 1: Get raw logits using forward()
    print("\n4. Getting raw logits...")
    logits = model.forward(dummy_audio, sampling_rate)
    print(f"   ✓ Logits shape: {logits.shape}")
    print(f"   ✓ Raw logits: {logits.cpu().detach().numpy()}")
    
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)
    print(f"   ✓ Probabilities: {probabilities.cpu().detach().numpy()}")
    
    # Method 2: Get direct prediction using predict()
    print("\n5. Getting direct prediction...")
    predicted_class = model.predict(dummy_audio, sampling_rate)
    print(f"   ✓ Predicted class ID: {predicted_class}")
    
    # Show confidence scores
    confidence_scores = probabilities.cpu().detach().numpy()[0]
    print(f"\n6. Confidence scores for each class:")
    for i, score in enumerate(confidence_scores):
        print(f"   Class {i}: {score:.4f}")
    
    print(f"\n   Most confident prediction: Class {predicted_class} ({confidence_scores[predicted_class]:.4f})")
    
    print("\n=== Example completed successfully! ===")

if __name__ == "__main__":
    main()
