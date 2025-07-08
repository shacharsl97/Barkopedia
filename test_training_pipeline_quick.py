#!/usr/bin/env python3
"""
Quick test of the training pipeline with a small dataset subset.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from barkopedia_datasets.simple_barkopedia_gender_dataset import create_simple_barkopedia_gender_dataset
from models.ast_classification_model import ASTClassificationModel
from models.wav2vec2_classification_model import Wav2Vec2ClassificationModel
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

def test_training_pipeline():
    """Test the complete training pipeline with a small subset."""
    
    print("=== Testing Training Pipeline (Small Subset) ===\n")
    
    # Load HuggingFace token if available
    hf_token = os.environ.get("HF_TOKEN")
    
    # Create dataset with segmentation disabled for faster testing
    print("1. Creating datasets...")
    splits = create_simple_barkopedia_gender_dataset(
        hf_token=hf_token,
        apply_cleaning=False,
        enable_segmentation=False,  # Disable segmentation for faster testing
        max_duration=10.0,  # Limit max duration
        min_duration=0.5
    )
    
    train_dataset = splits["train"]
    val_dataset = splits["validation"]
    
    # Create small subsets for testing
    train_subset = Subset(train_dataset, range(min(100, len(train_dataset))))  # Use first 100 samples
    val_subset = Subset(val_dataset, range(min(50, len(val_dataset))))  # Use first 50 samples
    
    print(f"Train subset: {len(train_subset)} samples")
    print(f"Validation subset: {len(val_subset)} samples")
    
    # Test data loading
    print("\n2. Testing data loading...")
    train_loader = train_dataset.get_dataloader(batch_size=4, shuffle=True, num_workers=0)
    val_loader = val_dataset.get_dataloader(batch_size=4, shuffle=False, num_workers=0)
    
    # Get a batch
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    print(f"Train batch input_values shape: {train_batch['input_values'].shape}")
    print(f"Train batch labels: {train_batch['labels']}")
    
    # Test model creation
    print("\n3. Testing model creation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test AST model
    ast_model = ASTClassificationModel(num_labels=2)
    ast_model.load_backbone({"model_name": "MIT/ast-finetuned-audioset-10-10-0.4593"})
    print(f"AST model created successfully")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    ast_outputs = ast_model.forward(train_batch['audio'].numpy(), 16000)
    print(f"AST outputs shape: {ast_outputs.shape}")
    print(f"AST outputs range: [{ast_outputs.min().item():.3f}, {ast_outputs.max().item():.3f}]")
    
    # Test training step
    print("\n5. Testing training step...")
    model = ast_model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()
    
    model.backbone.model.train()
    model.classifier.train()
    
    # Training step
    optimizer.zero_grad()
    
    with autocast():
        outputs = model.forward(train_batch['audio'].numpy(), 16000)
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
    
    # Test validation step
    print("\n6. Testing validation step...")
    model.backbone.model.eval()
    model.classifier.eval()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model.forward(batch['audio'].numpy(), 16000)
            labels = batch['labels'].to(device)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
    
    val_loss = total_loss / len(val_loader)
    val_accuracy = total_correct / total_samples
    
    print(f"Validation results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  Samples: {total_samples}")
    
    # Test a few training epochs
    print("\n7. Testing mini training loop...")
    model.backbone.model.train()
    model.classifier.train()
    
    for epoch in range(3):
        epoch_loss = 0
        epoch_correct = 0
        epoch_samples = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            with autocast():
                outputs = model.forward(batch['audio'].numpy(), 16000)
                labels = batch['labels'].to(device)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            
            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_samples += labels.size(0)
        
        epoch_loss /= len(train_loader)
        epoch_accuracy = epoch_correct / epoch_samples
        
        print(f"Epoch {epoch + 1}/3: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}")
    
    print("\n=== TRAINING PIPELINE TEST PASSED! ===")
    print("The training pipeline is working correctly.")
    return True

if __name__ == "__main__":
    success = test_training_pipeline()
    sys.exit(0 if success else 1)
