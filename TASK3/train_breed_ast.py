#!/usr/bin/env python3
"""
Training script for Barkopedia Dog Breed Classification (TASK3) using AST Classification Model.
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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from barkopedia_datasets import create_simple_barkopedia_breed_dataset, SimpleDatasetConfig
from models.ast_classification_model import ASTClassificationModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AST model for dog breed classification')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='MIT/ast-finetuned-audioset-10-10-0.4593',
                       help='Pre-trained AST model name or path')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for optimizer')
    
    # Dataset arguments
    parser.add_argument('--cache_dir', type=str, default='./barkopedia_breed_cache',
                       help='Directory to cache the dataset')
    parser.add_argument('--apply_cleaning', action='store_true',
                       help='Apply audio cleaning preprocessing')
    parser.add_argument('--max_duration', type=float, default=5.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--sampling_rate', type=int, default=16000,
                       help='Audio sampling rate')
    
    # Training configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--save_dir', type=str, default='./breed_classification_model',
                       help='Directory to save the trained model')
    parser.add_argument('--eval_steps', type=int, default=100,
                       help='Evaluation frequency in steps')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Model saving frequency in steps')
    
    # HuggingFace token
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace token for accessing datasets')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with limited data')
    
    return parser.parse_args()


def load_hf_token():
    """Load HuggingFace token from various sources."""
    try:
        hf_token_path = os.path.join(os.path.dirname(__file__), 'hf_token.py')
        if os.path.exists(hf_token_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("hf_token", hf_token_path)
            hf_token_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hf_token_module)
            return hf_token_module.HF_TOKEN
    except:
        pass
    return os.environ.get('HF_TOKEN')


def setup_device(device_arg):
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_datasets(args, hf_token):
    """Create train and test datasets."""
    logger.info("Loading datasets...")
    
    try:
        splits = create_simple_barkopedia_breed_dataset(
            cache_dir=args.cache_dir,
            hf_token=hf_token,
            apply_cleaning=args.apply_cleaning,
            sampling_rate=args.sampling_rate,
            max_duration=args.max_duration
        )
        
        train_dataset = splits['train']
        test_dataset = splits['test']
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Test dataset: {len(test_dataset)} samples")
        
        # Print class distribution
        train_dist = train_dataset.get_class_distribution()
        logger.info(f"Train class distribution: {train_dist}")
        
        if args.debug:
            train_dataset.data = train_dataset.data[:100]
            train_dataset.labels = train_dataset.labels[:100]
            test_dataset.data = test_dataset.data[:50]
            test_dataset.labels = test_dataset.labels[:50]
            logger.info("Debug mode: Limited dataset size")
        
        return train_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise


def create_model(args, device):
    """Create and initialize the AST classification model."""
    logger.info("Creating AST classification model...")
    
    # Barkopedia breed classification has 5 classes
    model = ASTClassificationModel(num_labels=5, device=device)
    
    backbone_args = {"model_name": args.model_name}
    model.load_backbone(backbone_args)
    
    logger.info(f"Model created with backbone: {args.model_name}")
    logger.info(f"Number of labels: 5 (dog breeds)")
    
    return model


def custom_collate_fn(batch):
    input_values = torch.stack([item['input_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return {
        'input_values': input_values,
        'labels': labels
    }


def evaluate_model(model, dataloader, device, label_names, sampling_rate=16000):
    model.backbone.model.eval()
    model.classifier.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            logits = model.forward(input_values.cpu().numpy(), sampling_rate)
            logits = logits.to(device)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    avg_loss = total_loss / len(dataloader)
    report = classification_report(all_labels, all_predictions, target_names=label_names, output_dict=True)
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'avg_loss': avg_loss,
        'classification_report': report,
        'predictions': all_predictions,
        'true_labels': all_labels
    }


def train_model(model, train_dataset, test_dataset, args, device):
    logger.info("Starting training...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    optimizer = optim.AdamW(
        model.classifier.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    eval_results = []
    best_f1 = 0.0
    # Example breed label names (replace with actual if available)
    label_names = ['breed_0', 'breed_1', 'breed_2', 'breed_3', 'breed_4']
    os.makedirs(args.save_dir, exist_ok=True)
    total_steps = 0
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        model.backbone.model.train()
        model.classifier.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for step, batch in enumerate(progress_bar):
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model.forward(input_values.cpu().numpy(), args.sampling_rate)
            logits = logits.to(device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_steps += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
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
                if eval_result['f1_score'] > best_f1:
                    best_f1 = eval_result['f1_score']
                    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
                    torch.save({
                        'classifier_state_dict': model.classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'f1_score': best_f1,
                        'step': total_steps,
                        'epoch': epoch + 1,
                        'args': vars(args)
                    }, best_model_path)
                    logger.info(f"New best model saved with F1: {best_f1:.4f}")
            if total_steps % args.save_steps == 0:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_step_{total_steps}.pth')
                torch.save({
                    'classifier_state_dict': model.classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': total_steps,
                    'epoch': epoch + 1,
                    'args': vars(args)
                }, checkpoint_path)
                logger.info(f"Checkpoint saved at step {total_steps}")
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        logger.info(f"Epoch {epoch + 1} completed - Average loss: {avg_epoch_loss:.4f}")
    logger.info("Running final evaluation...")
    final_eval = evaluate_model(model, test_loader, device, label_names, args.sampling_rate)
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save({
        'classifier_state_dict': model.classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_results': final_eval,
        'train_losses': train_losses,
        'eval_results': eval_results,
        'args': vars(args)
    }, final_model_path)
    results = {
        'final_evaluation': final_eval,
        'train_losses': train_losses,
        'evaluation_history': eval_results,
        'best_f1_score': best_f1,
        'training_args': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    results_path = os.path.join(args.save_dir, 'training_results.json')
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
    logger.info(f"Training completed!")
    logger.info(f"Final Accuracy: {final_eval['accuracy']:.4f}")
    logger.info(f"Final F1 Score: {final_eval['f1_score']:.4f}")
    logger.info(f"Best F1 Score: {best_f1:.4f}")
    logger.info(f"Results saved to: {args.save_dir}")
    return model, final_eval


def main():
    args = parse_arguments()
    device = setup_device(args.device)
    if args.hf_token is None:
        args.hf_token = load_hf_token()
    if args.hf_token is None:
        logger.warning("No HuggingFace token found. You may need to provide one for dataset access.")
    try:
        train_dataset, test_dataset = create_datasets(args, args.hf_token)
        model = create_model(args, device)
        trained_model, final_results = train_model(model, train_dataset, test_dataset, args, device)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
