#!/usr/bin/env python3
"""
Wav2Vec2 Model for Individual Dog Recognition (Task 4)
60-class classification for recognizing individual dogs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.wav2vec2_classification_model import Wav2Vec2ClassificationModel

class IndividualDogWav2Vec2Model(nn.Module):
    """
    Wav2Vec2 model for individual dog recognition.
    Designed for 60-class classification (60 individual dogs).
    """
    
    def __init__(self, num_dogs=60, device=None, pooling_mode="mean", 
                 dropout_rate=0.1, hidden_layers=[512, 256]):
        super().__init__()
        
        self.num_dogs = num_dogs
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.pooling_mode = pooling_mode
        self.dropout_rate = dropout_rate
        self.hidden_layers = hidden_layers
        
        # Wav2Vec2 base model has 768 hidden dimensions
        self.hidden_size = 768
        
        # Backbone will be loaded later
        self.backbone = None
        self.feature_extractor = None
        
        # Create a custom classifier that can handle single samples
        self.feature_projector = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, hidden_layers[0]),
            nn.ReLU()
        ).to(self.device)
        
        # Middle layers (if any)
        if len(hidden_layers) > 1:
            middle_layers = []
            for i in range(len(hidden_layers) - 1):
                middle_layers.extend([
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_layers[i], hidden_layers[i+1]),
                    nn.ReLU()
                ])
            self.middle_layers = nn.Sequential(*middle_layers).to(self.device)
        else:
            self.middle_layers = nn.Identity().to(self.device)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], num_dogs).to(self.device)
        
        # Additional pooling strategies for better feature extraction
        self.attention_pooling = AttentionPooling(self.hidden_size, device=self.device)
    
    def load_backbone(self, backbone_args: dict):
        """Load the Wav2Vec2 backbone model."""
        from models.wav2vec2_classification_model import Wav2Vec2Backbone
        self.backbone = Wav2Vec2Backbone(**backbone_args, device=self.device)
        self.backbone.load()
        self.feature_extractor = self.backbone.feature_extractor
    
    def _pool_features(self, features):
        """Enhanced pooling with multiple strategies."""
        batch_size, seq_len, hidden_dim = features.shape
        
        if self.pooling_mode == "mean":
            return torch.mean(features, dim=1)
        elif self.pooling_mode == "max":
            return torch.max(features, dim=1)[0]
        elif self.pooling_mode == "cls":
            return features[:, 0, :]
        elif self.pooling_mode == "attention":
            return self.attention_pooling(features)
        elif self.pooling_mode == "mean_max":
            # Combine mean and max pooling
            mean_pool = torch.mean(features, dim=1)
            max_pool = torch.max(features, dim=1)[0]
            return torch.cat([mean_pool, max_pool], dim=1)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")
    
    def forward(self, batch_data, device=None):
        """
        Forward pass for individual dog recognition.
        
        Args:
            batch_data: Dictionary or tensor containing audio data
            device: Device to run computation on
        
        Returns:
            Logits for 60 dog classes
        """
        if device is None:
            device = self.device
        
        # Make sure all components are on the correct device
        if next(self.feature_projector.parameters()).device != device:
            self.feature_projector = self.feature_projector.to(device)
            self.middle_layers = self.middle_layers.to(device)
            self.output_layer = self.output_layer.to(device)
        
        # Handle different input formats
        if isinstance(batch_data, dict):
            if 'audio' in batch_data:
                audio = batch_data['audio']
                sampling_rate = batch_data.get('sampling_rate', 16000)
            elif 'input_values' in batch_data:
                # Pre-processed input values
                inputs = {'input_values': batch_data['input_values'].to(device)}
                
                # Forward through Wav2Vec2 backbone
                if hasattr(self.backbone, 'model'):
                    with torch.no_grad():
                        outputs = self.backbone.model(**inputs, output_hidden_states=True)
                        features = outputs.last_hidden_state
                else:
                    raise ValueError("Backbone model not loaded")
                
                # Pool features and classify
                pooled_features = self._pool_features(features)
                
                # Custom forward pass through classifier components
                x = self.feature_projector(pooled_features)
                x = self.middle_layers(x)
                logits = self.output_layer(x)
                return logits
            else:
                raise ValueError("Batch must contain 'audio' or 'input_values'")
        else:
            # Assume it's raw audio tensor
            audio = batch_data
            sampling_rate = 16000
        
        # Process raw audio
        try:
            return super().forward(audio, sampling_rate)
        except AttributeError:
            # If super().forward is not available, process raw audio here
            audio_tensor = torch.tensor(audio).float().to(device)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
            inputs = self.feature_extractor(
                audio_tensor.cpu().numpy(),
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = self.backbone.model(**inputs, output_hidden_states=True)
                features = outputs.last_hidden_state
            
            pooled_features = self._pool_features(features)
            logits = self.classifier(pooled_features)
            return logits
    
    def predict_dog_id(self, audio, sampling_rate=16000, return_probabilities=False):
        """
        Predict dog ID from audio.
        
        Args:
            audio: Audio tensor or numpy array
            sampling_rate: Sampling rate
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Predicted dog ID (1-60) and optionally probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio, sampling_rate)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            
            # Convert back to 1-60 dog IDs
            dog_ids = predicted_class.cpu().numpy() + 1
            
            if return_probabilities:
                return dog_ids, probabilities.cpu().numpy()
            else:
                return dog_ids
    
    def compute_accuracy(self, outputs, targets):
        """
        Compute top-1 and top-5 accuracy metrics.
        
        Args:
            outputs: Model output logits
            targets: Ground truth labels
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Ensure outputs is logits, not raw features
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        batch_size = targets.size(0)
        
        # Top-1 accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        
        # Top-5 accuracy (important for 60-class problem)
        _, top5_indices = torch.topk(outputs, 5, dim=1)
        top5_correct = 0
        
        for i in range(batch_size):
            if targets[i] in top5_indices[i]:
                top5_correct += 1
        
        return {
            'top1_correct': correct,
            'top5_correct': top5_correct,
            'top1_accuracy': correct / batch_size,
            'top5_accuracy': top5_correct / batch_size,
            'total': batch_size
        }


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence features."""
    
    def __init__(self, hidden_dim, device=None):
        super().__init__()
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        ).to(self.device)
    
    def forward(self, features):
        # features: [batch_size, seq_len, hidden_dim]
        # Ensure the attention network is on the same device as the features
        device = features.device
        if next(self.attention.parameters()).device != device:
            self.attention = self.attention.to(device)
        
        attention_weights = self.attention(features)  # [batch_size, seq_len, 1]
        pooled = torch.sum(features * attention_weights, dim=1)  # [batch_size, hidden_dim]
        return pooled


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in dog recognition.
    Some dogs might have more training samples than others.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, num_classes=60):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_individual_dog_wav2vec2_model(num_dogs=60, device=None, 
                                        pooling_mode="attention", **kwargs):
    """
    Factory function to create Individual Dog Recognition Wav2Vec2 model.
    
    Args:
        num_dogs: Number of individual dogs (60)
        device: Device to run on
        pooling_mode: Pooling strategy ("mean", "max", "attention", "mean_max")
        **kwargs: Additional arguments
    
    Returns:
        IndividualDogWav2Vec2Model instance
    """
    return IndividualDogWav2Vec2Model(
        num_dogs=num_dogs,
        device=device,
        pooling_mode=pooling_mode,
        **kwargs
    )


if __name__ == "__main__":
    """Test the Individual Dog Recognition model."""
    
    print("=== Testing Individual Dog Recognition Wav2Vec2 Model ===\n")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_individual_dog_wav2vec2_model(
        num_dogs=60,
        device=device,
        pooling_mode="attention"
    )
    
    # Load backbone
    print("Loading Wav2Vec2 backbone...")
    model.load_backbone({'model_name': 'facebook/wav2vec2-base'})
    
    # Test with dummy batch
    print("\n1. Testing forward pass...")
    batch_size = 4
    sequence_length = 16000  # 1 second of audio
    
    # Test with input_values (pre-processed)
    dummy_batch = {
        'input_values': torch.randn(batch_size, sequence_length).to(device),
        'labels': torch.randint(0, 60, (batch_size,)).to(device),
        'sampling_rate': 16000
    }
    
    # Forward pass
    outputs = model.forward(dummy_batch)
    print(f"   ✓ Output logits shape: {outputs.shape}")
    print(f"   ✓ Expected shape: [batch_size={batch_size}, num_dogs=60]")
    
    # Test accuracy computation
    print("\n2. Testing accuracy computation...")
    accuracy_metrics = model.compute_accuracy(outputs, dummy_batch['labels'])
    print(f"   ✓ Top-1 accuracy: {accuracy_metrics['top1_accuracy']:.4f}")
    print(f"   ✓ Top-5 accuracy: {accuracy_metrics['top5_accuracy']:.4f}")
    
    # Test dog ID prediction
    print("\n3. Testing dog ID prediction...")
    dummy_audio = torch.randn(16000).to(device)  # 1 second
    predicted_dog_ids, probabilities = model.predict_dog_id(
        dummy_audio, return_probabilities=True
    )
    print(f"   ✓ Predicted dog ID: {predicted_dog_ids[0]} (range: 1-60)")
    print(f"   ✓ Top 3 probabilities: {np.sort(probabilities[0])[-3:]}")
    
    # Test loss function
    print("\n4. Testing Focal Loss...")
    focal_loss = FocalLoss(num_classes=60)
    loss_value = focal_loss(outputs, dummy_batch['labels'])
    print(f"   ✓ Focal loss: {loss_value.item():.4f}")
    
    # Model size information
    print("\n5. Model information...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    
    print("\n=== All tests passed! ===")
