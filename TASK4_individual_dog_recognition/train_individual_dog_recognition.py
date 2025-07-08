#!/usr/bin/env python3
"""
Training Script for Individual Dog Recognition (Task 4)
Train Wav2Vec2 model to recognize 60 individual dogs
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TASK4_individual_dog_recognition.individual_dog_dataset import create_individual_dog_dataset
from TASK4_individual_dog_recognition.individual_dog_wav2vec2_model import (
    create_individual_dog_wav2vec2_model, FocalLoss
)

class IndividualDogTrainer:
    """Trainer for Individual Dog Recognition task."""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function - use Focal Loss for potential class imbalance
        if config.use_focal_loss:
            self.criterion = FocalLoss(
                alpha=config.focal_alpha,
                gamma=config.focal_gamma,
                num_classes=60
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
        # Learning rate scheduler
        if config.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs
            )
        elif config.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config.step_size, gamma=config.gamma
            )
        elif config.scheduler == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=config.patience, factor=0.5
            )
        else:
            self.scheduler = None
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler() if config.use_mixed_precision else None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_top1_accs = []
        self.val_top1_accs = []
        self.train_top5_accs = []
        self.val_top5_accs = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        total_top1_correct = 0
        total_top5_correct = 0
        total_samples = 0
        
        # Initialize progress bar for training
        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch}/{self.config.epochs}",
            leave=True,  # Changed from False to True for better visibility
            ncols=100    # Slightly wider for better formatting
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model.forward(batch)
                    loss = self.criterion(outputs, batch['labels'])
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model.forward(batch)
                loss = self.criterion(outputs, batch['labels'])
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            
            # Compute accuracy
            accuracy_metrics = self.model.compute_accuracy(outputs, batch['labels'])
            total_top1_correct += accuracy_metrics['top1_correct']
            total_top5_correct += accuracy_metrics['top5_correct']
            total_samples += accuracy_metrics['total']
            
            # Update progress bar with current metrics - keep it simple
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'top1': f"{total_top1_correct / total_samples:.4f}", 
            })
        
        # Epoch statistics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_top1_acc = total_top1_correct / total_samples
        epoch_top5_acc = total_top5_correct / total_samples
        
        return epoch_loss, epoch_top1_acc, epoch_top5_acc
    
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        total_top1_correct = 0
        total_top5_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        # Initialize progress bar for validation
        pbar = tqdm(
            self.val_loader,
            desc=f"Val Epoch {epoch}/{self.config.epochs}",
            leave=True,  # Changed from False to True for better visibility
            ncols=100    # Slightly wider for better formatting
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model.forward(batch)
                loss = self.criterion(outputs, batch['labels'])
                
                running_loss += loss.item()
                
                # Compute accuracy
                accuracy_metrics = self.model.compute_accuracy(outputs, batch['labels'])
                total_top1_correct += accuracy_metrics['top1_correct']
                total_top5_correct += accuracy_metrics['top5_correct']
                total_samples += accuracy_metrics['total']
                
                # Store predictions for detailed analysis
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                
                # Update progress bar - simplified to avoid cluttering
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'top1': f"{total_top1_correct / total_samples:.4f}", 
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_top1_acc = total_top1_correct / total_samples
        epoch_top5_acc = total_top5_correct / total_samples
        
        return epoch_loss, epoch_top1_acc, epoch_top5_acc, all_predictions, all_labels
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.epochs} epochs...")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\n{'='*30} Epoch {epoch}/{self.config.epochs} {'='*30}")
            
            # Training
            train_loss, train_top1_acc, train_top5_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_top1_acc, val_top5_acc, val_preds, val_labels = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_top1_acc)
                else:
                    self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_top1_accs.append(train_top1_acc)
            self.val_top1_accs.append(val_top1_acc)
            self.train_top5_accs.append(train_top5_acc)
            self.val_top5_accs.append(val_top5_acc)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary - only print after epoch is completed
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Training:   Loss: {train_loss:.6f}, Top1: {train_top1_acc:.4f}, Top5: {train_top5_acc:.4f}")
            print(f"  Validation: Loss: {val_loss:.6f}, Top1: {val_top1_acc:.4f}, Top5: {val_top5_acc:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")
            
            # Print per-class evaluation metrics for this epoch
            if epoch % self.config.eval_interval == 0 or epoch == 1 or epoch == self.config.epochs:
                # Create a mini classification report for the current epoch
                report = classification_report(
                    val_labels, val_preds,
                    target_names=[f"Dog_{i+1}" for i in range(60)],
                    output_dict=True
                )
                
                # Sort dog classes by F1 score and display top 5 and bottom 5
                dog_f1_scores = {f"Dog_{i+1}": report[f"Dog_{i+1}"]["f1-score"] for i in range(60)}
                top_dogs = sorted(dog_f1_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                bottom_dogs = sorted(dog_f1_scores.items(), key=lambda x: x[1])[:5]
                
                print("\nClass Performance (F1 Scores)")
                print(f"  {'Top 5:':<10}", end="")
                print(", ".join([f"{dog}={f1:.3f}" for dog, f1 in top_dogs]))
                print(f"  {'Bottom 5:':<10}", end="")
                print(", ".join([f"{dog}={f1:.3f}" for dog, f1 in bottom_dogs]))
                print(f"  {'Overall:':<10} Macro F1={report['macro avg']['f1-score']:.4f}, Weighted F1={report['weighted avg']['f1-score']:.4f}")
            
            # Save best model with concise message
            if val_top1_acc > self.best_val_acc:
                self.best_val_acc = val_top1_acc
                self.best_epoch = epoch
                self.save_model(f"{self.config.model_save_path}/best_model.pth")
                print(f"  ✓ New best model saved! Val Acc: {val_top1_acc:.4f}")
            
            # Save checkpoint without unnecessary message
            if epoch % self.config.save_interval == 0:
                checkpoint_path = f"{self.config.model_save_path}/checkpoint_epoch_{epoch}.pth"
                self.save_model(checkpoint_path)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
            
            # Early stopping
            if hasattr(self.config, 'early_stopping_patience'):
                if epoch - self.best_epoch >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        
        # Generate final plots and analysis
        self.plot_training_curves()
        self.generate_classification_report(val_preds, val_labels)
    
    def save_model(self, filepath):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'epoch': len(self.train_losses),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_top1_accs': self.train_top1_accs,
            'val_top1_accs': self.val_top1_accs,
            'train_top5_accs': self.train_top5_accs,
            'val_top5_accs': self.val_top5_accs,
        }
        
        torch.save(checkpoint, filepath)
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Top-1 Accuracy curves
        axes[0, 1].plot(self.train_top1_accs, label='Train Top-1 Acc')
        axes[0, 1].plot(self.val_top1_accs, label='Validation Top-1 Acc')
        axes[0, 1].set_title('Top-1 Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-5 Accuracy curves
        axes[1, 0].plot(self.train_top5_accs, label='Train Top-5 Acc')
        axes[1, 0].plot(self.val_top5_accs, label='Validation Top-5 Acc')
        axes[1, 0].set_title('Top-5 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.learning_rates)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.model_save_path}/training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_classification_report(self, predictions, labels):
        """Generate detailed classification report."""
        # Classification report
        report = classification_report(
            labels, predictions,
            target_names=[f"Dog_{i+1}" for i in range(60)],
            output_dict=True
        )
        
        # Save report
        with open(f"{self.config.model_save_path}/classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nClassification Report Summary:")
        print(f"  Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
        print(f"  Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Individual Dog Recognition Model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--pooling_mode', type=str, default='attention', 
                       choices=['mean', 'max', 'attention', 'mean_max'], help='Pooling strategy')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'step', 'reduce_on_plateau', 'none'])
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--model_save_path', type=str, default='./models/individual_dog_recognition',
                       help='Path to save models')
    parser.add_argument('--log_interval', type=int, default=50, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Model save interval')
    parser.add_argument('--enable_segmentation', action='store_true', help='Enable audio segmentation')
    parser.add_argument('--segment_duration', type=float, default=2.0, help='Segment duration in seconds')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")
    
    # Create datasets
    print("Loading datasets...")
    datasets = create_individual_dog_dataset(
        hf_token=hf_token,
        apply_cleaning=True,
        enable_segmentation=args.enable_segmentation,
        segment_duration=args.segment_duration,
        sampling_rate=16000
    )
    
    # Create data loaders
    train_loader = datasets['train'].get_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = datasets['validation'].get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_individual_dog_wav2vec2_model(
        num_dogs=60,
        device=device,
        pooling_mode=args.pooling_mode
    )
    
    # Load Wav2Vec2 backbone
    model.load_backbone({'model_name': 'facebook/wav2vec2-base'})
    
    # Create trainer
    trainer = IndividualDogTrainer(model, train_loader, val_loader, device, args)
    
    # Start training
    trainer.train()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
