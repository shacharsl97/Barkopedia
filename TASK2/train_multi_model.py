#!/usr/bin/env python3
"""
Multi-Model Training Script for Barkopedia Gender Classification.
Supports AST, Wav2Vec2 Custom, and Wav2Vec2 Pretrained models.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from barkopedia_datasets import create_simple_barkopedia_gender_dataset, SimpleDatasetConfig
from models.ast_classification_model import ASTClassificationModel
from models.wav2vec2_classification_model import create_wav2vec2_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train models for gender classification')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, choices=['ast', 'wav2vec2_custom', 'wav2vec2_pretrained'], 
                       default='ast', help='Type of model to train')
    parser.add_argument('--model_name', type=str, default='MIT/ast-finetuned-audioset-10-10-0.4593',
                       help='Pre-trained model name or path')
    parser.add_argument('--pooling_mode', type=str, choices=['mean', 'max', 'cls'], default='mean',
                       help='Pooling mode for Wav2Vec2 custom model')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for optimizer')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Number of warmup steps for learning rate scheduler')
    
    # Dataset arguments
    parser.add_argument('--cache_dir', type=str, default='./barkopedia_gender_cache',
                       help='Directory to cache the dataset')
    parser.add_argument('--apply_cleaning', action='store_true',
                       help='Apply audio cleaning preprocessing')
    parser.add_argument('--max_duration', type=float, default=5.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--sampling_rate', type=int, default=16000,
                       help='Audio sampling rate')
    
    # Training configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cuda:0, cuda:1, cpu, or auto)')
    parser.add_argument('--save_dir', type=str, default='./multi_model_training',
                       help='Directory to save the trained model')
    parser.add_argument('--eval_steps', type=int, default=400,
                       help='Evaluation frequency in steps')
    parser.add_argument('--save_steps', type=int, default=400,
                       help='Model saving frequency in steps')
    parser.add_argument('--log_steps', type=int, default=50,
                       help='Logging frequency in steps')
    
    # HuggingFace token
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace token for accessing datasets')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with limited data')
    
    # Segmentation arguments
    parser.add_argument('--enable_segmentation', action='store_true',
                       help='Enable automatic audio segmentation into chunks')
    parser.add_argument('--segment_duration', type=float, default=2.0,
                       help='Target duration for each segment (0.3-5.0 seconds)')
    parser.add_argument('--segment_overlap', type=float, default=0.1,
                       help='Overlap between segments in seconds')
    parser.add_argument('--energy_threshold', type=float, default=0.01,
                       help='Energy threshold for silence detection (0.001-0.1)')
    
    # Visualization arguments
    parser.add_argument('--create_plots', action='store_true', default=True,
                       help='Create training plots after completion')
    parser.add_argument('--plot_style', type=str, default='seaborn-v0_8',
                       help='Matplotlib style for plots')
    
    return parser.parse_args()


def load_hf_token():
    """Load HuggingFace token from various sources."""
    # Try to load from hf_token.py file
    try:
        import importlib.util
        hf_token_path = os.path.join(os.path.dirname(__file__), 'hf_token.py')
        if os.path.exists(hf_token_path):
            spec = importlib.util.spec_from_file_location("hf_token", hf_token_path)
            hf_token_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hf_token_module)
            return hf_token_module.HF_TOKEN
    except:
        pass
    
    # Try environment variable
    return os.environ.get('HF_TOKEN')


def setup_device(device_arg):
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_arg.isdigit():
        # Handle numeric device IDs like '0', '1', '2' -> 'cuda:0', 'cuda:1', 'cuda:2'
        device = torch.device(f'cuda:{device_arg}')
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        gpu_id = device.index if device.index is not None else 0
        logger.info(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
    
    return device


def create_datasets(args, hf_token):
    """Create train and test datasets."""
    logger.info("Loading datasets...")
    
    try:
        splits = create_simple_barkopedia_gender_dataset(
            cache_dir=args.cache_dir,
            hf_token=hf_token,
            apply_cleaning=args.apply_cleaning,
            sampling_rate=args.sampling_rate,
            max_duration=args.max_duration,
            enable_segmentation=args.enable_segmentation,
            segment_duration=args.segment_duration,
            segment_overlap=args.segment_overlap,
            energy_threshold=args.energy_threshold
        )
        
        train_dataset = splits['train']
        validation_dataset = splits['validation']
        test_dataset = splits['test']
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Validation dataset: {len(validation_dataset)} samples")
        logger.info(f"Test dataset: {len(test_dataset)} samples")
        
        # Print class distribution
        train_dist = train_dataset.get_class_distribution()
        validation_dist = validation_dataset.get_class_distribution()
        logger.info(f"Train class distribution: {train_dist}")
        logger.info(f"Validation class distribution: {validation_dist}")
        logger.info(f"Test dataset loaded for inference (no labels)")
        
        if args.debug:
            # Limit dataset size for debugging
            train_dataset.data = train_dataset.data[:200]
            train_dataset.labels = train_dataset.labels[:200]
            validation_dataset.data = validation_dataset.data[:100]
            validation_dataset.labels = validation_dataset.labels[:100]
            test_dataset.data = test_dataset.data[:50]
            test_dataset.labels = test_dataset.labels[:50]
            logger.info(f"Debug mode: Limited dataset size - Train: {len(train_dataset)}, Validation: {len(validation_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, validation_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise


def create_model(args, device, num_labels):
    """Create and initialize the specified model."""
    logger.info(f"Creating {args.model_type} model...")
    
    if args.model_type == 'ast':
        model = ASTClassificationModel(num_labels=num_labels, device=device)
        backbone_args = {"model_name": args.model_name}
        model.load_backbone(backbone_args)
        
    elif args.model_type == 'wav2vec2_custom':
        model = create_wav2vec2_model(
            num_labels=num_labels, 
            device=device, 
            model_type="custom",
            pooling_mode=args.pooling_mode
        )
        backbone_args = {"model_name": args.model_name}
        model.load_backbone(backbone_args)
        
    elif args.model_type == 'wav2vec2_pretrained':
        model = create_wav2vec2_model(
            num_labels=num_labels, 
            device=device, 
            model_type="pretrained"
        )
        backbone_args = {"model_name": args.model_name}
        model.load_backbone(backbone_args)
        
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    logger.info(f"Model created with backbone: {args.model_name}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Number of labels: {num_labels}")
    
    return model


def custom_collate_fn(batch):
    """Custom collate function to handle batching."""
    audio_arrays = []
    labels = []
    sampling_rates = []
    
    for item in batch:
        audio = item['audio']
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32)
        
        audio_arrays.append(audio)
        labels.append(item['labels'])
        sampling_rates.append(item['sampling_rate'])
    
    assert all(sr == sampling_rates[0] for sr in sampling_rates), "All audio must have the same sampling rate"
    
    return {
        'audio': audio_arrays,
        'labels': torch.tensor(labels, dtype=torch.long),
        'sampling_rate': sampling_rates[0]
    }


def evaluate_model(model, dataloader, device, label_names, sampling_rate=16000):
    """Evaluate the model on a dataset."""
    # Set model to eval mode based on model type
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'model'):
        model.backbone.model.eval()
    if hasattr(model, 'classifier') and model.classifier is not None:
        model.classifier.eval()
    if hasattr(model, 'model'):
        model.model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio_arrays = batch['audio']
            labels = batch['labels'].to(device)
            sampling_rate = batch['sampling_rate']
            
            # Forward pass
            batch_logits = []
            for audio in audio_arrays:
                if len(audio.shape) > 1:
                    audio = audio.flatten()
                audio = audio.astype(np.float32)
                
                # Ensure minimum length
                min_length = 400
                if len(audio) < min_length:
                    padding_needed = min_length - len(audio)
                    audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)
                
                logits = model.forward(audio, sampling_rate)
                batch_logits.append(logits)
            
            logits = torch.cat(batch_logits, dim=0).to(device)
            
            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    avg_loss = total_loss / len(dataloader)
    
    logger.info(f"Evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Loss: {avg_loss:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'avg_loss': avg_loss,
        'predictions': all_predictions,
        'true_labels': all_labels
    }


def create_plots(train_losses, eval_results, save_dir, model_type):
    """Create training plots."""
    logger.info("Creating training plots...")
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_type.upper()} Model Training Results', fontsize=16)
        
        # Plot 1: Training Loss
        if train_losses:
            axes[0, 0].plot(train_losses, label='Training Loss', color='red', linewidth=2)
            axes[0, 0].set_title('Training Loss Over Time')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Evaluation Accuracy
        if eval_results:
            steps = [result['step'] for result in eval_results]
            accuracies = [result['accuracy'] for result in eval_results]
            axes[0, 1].plot(steps, accuracies, label='Validation Accuracy', color='blue', linewidth=2, marker='o')
            axes[0, 1].set_title('Validation Accuracy Over Steps')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Evaluation F1 Score
        if eval_results:
            f1_scores = [result['f1_score'] for result in eval_results]
            axes[1, 0].plot(steps, f1_scores, label='Validation F1 Score', color='green', linewidth=2, marker='s')
            axes[1, 0].set_title('Validation F1 Score Over Steps')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Evaluation Loss
        if eval_results:
            eval_losses = [result['avg_loss'] for result in eval_results]
            axes[1, 1].plot(steps, eval_losses, label='Validation Loss', color='orange', linewidth=2, marker='^')
            axes[1, 1].set_title('Validation Loss Over Steps')
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(save_dir, f'{model_type}_training_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to: {plot_path}")
        
        # Create a separate detailed metrics plot
        if eval_results:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            ax2 = ax.twinx()
            ax3 = ax.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            
            # Plot accuracy and F1 on left axis
            line1 = ax.plot(steps, accuracies, 'b-', label='Accuracy', linewidth=2, marker='o')
            line2 = ax.plot(steps, f1_scores, 'g-', label='F1 Score', linewidth=2, marker='s')
            
            # Plot loss on right axis
            line3 = ax2.plot(steps, eval_losses, 'r-', label='Validation Loss', linewidth=2, marker='^')
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Accuracy / F1 Score', color='black')
            ax2.set_ylabel('Loss', color='red')
            
            ax.tick_params(axis='y', labelcolor='black')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='center right')
            
            ax.set_title(f'{model_type.upper()} Model - Detailed Training Metrics')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            detailed_plot_path = os.path.join(save_dir, f'{model_type}_detailed_metrics.png')
            plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Detailed metrics plot saved to: {detailed_plot_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create plots: {e}")


def train_model(model, train_dataset, validation_dataset, args, device):
    """Train the model."""
    logger.info("Starting training...")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    test_loader = DataLoader(
        validation_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    # Setup optimizer and loss function
    if args.model_type == 'ast':
        optimizer = optim.AdamW(
            model.classifier.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
    elif args.model_type == 'wav2vec2_custom':
        optimizer = optim.AdamW(
            model.classifier.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
    elif args.model_type == 'wav2vec2_pretrained':
        optimizer = optim.AdamW(
            model.model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
    
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=args.num_epochs,
        pct_start=args.warmup_steps / (len(train_loader) * args.num_epochs)
    )
    
    # Training tracking
    train_losses = []
    eval_results = []
    best_f1 = 0.0
    label_names = list(train_dataset.id_to_label.values())
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    total_steps = 0
    epoch_losses = []
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Training phase
        if args.model_type == 'ast':
            if hasattr(model, 'backbone'):
                model.backbone.model.train()
            if hasattr(model, 'classifier'):
                model.classifier.train()
        elif args.model_type == 'wav2vec2_custom':
            if hasattr(model, 'backbone'):
                model.backbone.model.train()
            if hasattr(model, 'classifier'):
                model.classifier.train()
        elif args.model_type == 'wav2vec2_pretrained':
            if hasattr(model, 'model'):
                model.model.train()
        
        epoch_loss = 0
        step_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            audio_arrays = batch['audio']
            labels = batch['labels'].to(device)
            sampling_rate = batch['sampling_rate']
            
            optimizer.zero_grad()
            
            # Forward pass
            batch_logits = []
            for audio in audio_arrays:
                if len(audio.shape) > 1:
                    audio = audio.flatten()
                audio = audio.astype(np.float32)
                
                min_length = 400
                if len(audio) < min_length:
                    padding_needed = min_length - len(audio)
                    audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)
                
                logits = model.forward(audio, sampling_rate)
                batch_logits.append(logits)
            
            logits = torch.cat(batch_logits, dim=0).to(device)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            step_losses.append(loss.item())
            total_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Logging
            if total_steps % args.log_steps == 0:
                avg_recent_loss = np.mean(step_losses[-args.log_steps:])
                logger.info(f"Step {total_steps} - Loss: {avg_recent_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Evaluation
            if total_steps % args.eval_steps == 0:
                logger.info("Running evaluation...")
                eval_result = evaluate_model(model, test_loader, device, label_names, args.sampling_rate)
                eval_results.append({
                    'step': total_steps,
                    'epoch': epoch + 1,
                    **eval_result
                })
                
                logger.info(f"Step {total_steps} - "
                          f"Accuracy: {eval_result['accuracy']:.4f}, "
                          f"F1: {eval_result['f1_score']:.4f}, "
                          f"Loss: {eval_result['avg_loss']:.4f}")
                
                # Save best model
                if eval_result['f1_score'] > best_f1:
                    best_f1 = eval_result['f1_score']
                    best_model_path = os.path.join(args.save_dir, f'best_{args.model_type}_model.pth')
                    
                    save_dict = {
                        'step': total_steps,
                        'epoch': epoch + 1,
                        'f1_score': best_f1,
                        'args': vars(args)
                    }
                    
                    if args.model_type == 'ast':
                        save_dict['classifier_state_dict'] = model.classifier.state_dict()
                    elif args.model_type == 'wav2vec2_custom':
                        save_dict['classifier_state_dict'] = model.classifier.state_dict()
                    elif args.model_type == 'wav2vec2_pretrained':
                        save_dict['model_state_dict'] = model.model.state_dict()
                    
                    torch.save(save_dict, best_model_path)
                    logger.info(f"New best model saved with F1: {best_f1:.4f}")
                
                # Switch back to training mode
                if args.model_type == 'ast':
                    if hasattr(model, 'backbone'):
                        model.backbone.model.train()
                    if hasattr(model, 'classifier'):
                        model.classifier.train()
                elif args.model_type == 'wav2vec2_custom':
                    if hasattr(model, 'backbone'):
                        model.backbone.model.train()
                    if hasattr(model, 'classifier'):
                        model.classifier.train()
                elif args.model_type == 'wav2vec2_pretrained':
                    if hasattr(model, 'model'):
                        model.model.train()
            
            # Save checkpoint
            if total_steps % args.save_steps == 0:
                checkpoint_path = os.path.join(args.save_dir, f'{args.model_type}_checkpoint_step_{total_steps}.pth')
                
                save_dict = {
                    'step': total_steps,
                    'epoch': epoch + 1,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'args': vars(args)
                }
                
                if args.model_type == 'ast':
                    save_dict['classifier_state_dict'] = model.classifier.state_dict()
                elif args.model_type == 'wav2vec2_custom':
                    save_dict['classifier_state_dict'] = model.classifier.state_dict()
                elif args.model_type == 'wav2vec2_pretrained':
                    save_dict['model_state_dict'] = model.model.state_dict()
                
                torch.save(save_dict, checkpoint_path)
                logger.info(f"Checkpoint saved at step {total_steps}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        epoch_losses.append(avg_epoch_loss)
        logger.info(f"Epoch {epoch + 1} completed - Average loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_eval = evaluate_model(model, test_loader, device, label_names, args.sampling_rate)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f'final_{args.model_type}_model.pth')
    final_save_dict = {
        'final_results': final_eval,
        'train_losses': train_losses,
        'eval_results': eval_results,
        'args': vars(args)
    }
    
    if args.model_type == 'ast':
        final_save_dict['classifier_state_dict'] = model.classifier.state_dict()
    elif args.model_type == 'wav2vec2_custom':
        final_save_dict['classifier_state_dict'] = model.classifier.state_dict()
    elif args.model_type == 'wav2vec2_pretrained':
        final_save_dict['model_state_dict'] = model.model.state_dict()
    
    torch.save(final_save_dict, final_model_path)
    
    # Save training results
    results = {
        'final_evaluation': final_eval,
        'train_losses': train_losses,
        'evaluation_history': eval_results,
        'best_f1_score': best_f1,
        'training_args': vars(args),
        'model_type': args.model_type,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(args.save_dir, f'{args.model_type}_training_results.json')
    with open(results_path, 'w') as f:
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        json.dump(convert_numpy(results), f, indent=2)
    
    # Create plots
    if args.create_plots:
        create_plots(train_losses, eval_results, args.save_dir, args.model_type)
    
    logger.info(f"Training completed!")
    logger.info(f"Final Accuracy: {final_eval['accuracy']:.4f}")
    logger.info(f"Final F1 Score: {final_eval['f1_score']:.4f}")
    logger.info(f"Best F1 Score: {best_f1:.4f}")
    logger.info(f"Results saved to: {args.save_dir}")
    
    return model, final_eval


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup device
    device = setup_device(args.device)
    
    # Load HuggingFace token
    if args.hf_token is None:
        args.hf_token = load_hf_token()
    
    if args.hf_token is None:
        logger.warning("No HuggingFace token found. You may need to provide one for dataset access.")
    
    # Create model-specific save directory
    model_save_dir = os.path.join(args.save_dir, f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    args.save_dir = model_save_dir
    
    try:
        # Create datasets
        train_dataset, validation_dataset, test_dataset = create_datasets(args, args.hf_token)
        
        # Create model
        num_labels = len(train_dataset.id_to_label)
        model = create_model(args, device, num_labels)
        
        # Train model
        trained_model, final_results = train_model(model, train_dataset, validation_dataset, args, device)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
