#!/usr/bin/env python3
"""
Test script to verify the full training pipeline works correctly.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from barkopedia_datasets.simple_barkopedia_gender_dataset import SimpleBarkopediaGenderDataset
from barkopedia_datasets.simple_dataset_interface import SimpleDatasetConfig
from models.ast_classification_model import ASTClassificationModel
from models.wav2vec2_classification_model import Wav2Vec2ClassificationModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

def test_training_pipeline():
    """Test the complete training pipeline."""
    
    print("=== Testing Complete Training Pipeline ===\n")
    
    # Configuration
    config = SimpleDatasetConfig(
        dataset_name="Barkopedia",
        sampling_rate=16000,
        enable_segmentation=True,
        segment_duration=3.0,
        segment_overlap=0.5,
        min_segment_duration=1.0,
        max_segment_duration=5.0,
        max_duration=30.0,
        min_duration=0.5,
        apply_cleaning=True,
        cache_dir="./cache"
    )
    
    print(f"Dataset config: {config}")
    
    # Create datasets
    print("\n1. Creating datasets...")
    try:
        train_dataset = SimpleBarkopediaGenderDataset(config, split="train")
        val_dataset = SimpleBarkopediaGenderDataset(config, split="validation")
        
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Validation dataset: {len(val_dataset)} samples")
        
        # Test class distributions
        print(f"Train class distribution: {train_dataset.get_class_distribution()}")
        print(f"Validation class distribution: {val_dataset.get_class_distribution()}")
        
    except Exception as e:
        print(f"ERROR: Failed to create datasets: {e}")
        return False
    
    # Test data loading
    print("\n2. Testing data loading...")
    try:
        train_loader = train_dataset.get_dataloader(
            batch_size=4,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = val_dataset.get_dataloader(
            batch_size=4,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Get a batch
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        print(f"Train batch keys: {train_batch.keys()}")
        print(f"Train batch shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in train_batch.items()]}")
        print(f"Train batch labels: {train_batch['labels']}")
        
        print(f"Val batch keys: {val_batch.keys()}")
        print(f"Val batch shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in val_batch.items()]}")
        print(f"Val batch labels: {val_batch['labels']}")
        
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return False
    
    # Test model creation
    print("\n3. Testing model creation...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Test AST model
        ast_model = ASTClassificationModel(num_classes=2, model_name="MIT/ast-finetuned-audioset-10-10-0.4593")
        ast_model = ast_model.to(device)
        print(f"AST model created successfully")
        
        # Test Wav2Vec2 model
        wav2vec2_model = Wav2Vec2ClassificationModel(num_classes=2, model_name="facebook/wav2vec2-base")
        wav2vec2_model = wav2vec2_model.to(device)
        print(f"Wav2Vec2 model created successfully")
        
    except Exception as e:
        print(f"ERROR: Failed to create models: {e}")
        return False
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    try:
        # AST forward pass
        ast_outputs = ast_model(train_batch, device)
        print(f"AST outputs shape: {ast_outputs.shape}")
        print(f"AST outputs range: [{ast_outputs.min().item():.3f}, {ast_outputs.max().item():.3f}]")
        
        # Wav2Vec2 forward pass  
        wav2vec2_outputs = wav2vec2_model(train_batch, device)
        print(f"Wav2Vec2 outputs shape: {wav2vec2_outputs.shape}")
        print(f"Wav2Vec2 outputs range: [{wav2vec2_outputs.min().item():.3f}, {wav2vec2_outputs.max().item():.3f}]")
        
    except Exception as e:
        print(f"ERROR: Failed in forward pass: {e}")
        return False
    
    # Test training step
    print("\n5. Testing training step...")
    try:
        # Use AST model for training test
        model = ast_model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = GradScaler()
        
        model.train()
        
        # Training step
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(train_batch, device)
            labels = train_batch['labels'].to(device)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        
        print(f"Training step successful:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Predictions: {predicted.cpu().numpy()}")
        print(f"  Ground truth: {labels.cpu().numpy()}")
        
    except Exception as e:
        print(f"ERROR: Failed in training step: {e}")
        return False
    
    # Test validation step
    print("\n6. Testing validation step...")
    try:
        model.eval()
        
        with torch.no_grad():
            outputs = model(val_batch, device)
            labels = val_batch['labels'].to(device)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)
            
            print(f"Validation step successful:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Predictions: {predicted.cpu().numpy()}")
            print(f"  Ground truth: {labels.cpu().numpy()}")
        
    except Exception as e:
        print(f"ERROR: Failed in validation step: {e}")
        return False
    
    print("\n=== ALL TESTS PASSED! ===")
    print("The training pipeline is working correctly.")
    return True

if __name__ == "__main__":
    success = test_training_pipeline()
    sys.exit(0 if success else 1)
