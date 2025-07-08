#!/usr/bin/env python3
"""
Enhanced Wav2Vec2 model with dog identity awareness and contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from models.wav2vec2_classification_model import Wav2Vec2ClassificationModel

class DogAwareWav2Vec2Model(Wav2Vec2ClassificationModel):
    """
    Enhanced Wav2Vec2 model that leverages dog identity information.
    Implements contrastive learning and multi-task learning.
    """
    
    def __init__(self, num_labels=2, num_dogs=None, device=None, 
                 pooling_mode="mean", use_contrastive=True, use_dog_identity=False):
        super().__init__(num_labels, device, pooling_mode)
        
        self.num_dogs = num_dogs
        self.use_contrastive = use_contrastive
        self.use_dog_identity = use_dog_identity
        
        # Dog identity classification head (if enabled)
        if self.use_dog_identity and num_dogs:
            self.dog_classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size // 2, num_dogs)
            ).to(self.device)
        
        # Contrastive learning projection head
        if self.use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, 128),  # 128-dim embeddings for contrastive
                nn.L2Norm(dim=1)  # L2 normalize for cosine similarity
            ).to(self.device)
    
    def forward(self, batch: Dict, return_embeddings=False):
        """
        Enhanced forward pass with optional contrastive learning.
        
        Args:
            batch: Dictionary containing:
                - 'audio': Raw audio tensor
                - 'labels': Gender labels (0/1)
                - 'dog_ids': Dog identity labels (optional)
                - 'sampling_rate': Sampling rate
            return_embeddings: Whether to return feature embeddings
        
        Returns:
            Dictionary with logits, embeddings, and optional dog predictions
        """
        audio = batch['audio']
        sampling_rate = batch.get('sampling_rate', 16000)
        
        # Extract features using backbone
        features = self.backbone.forward(audio, sampling_rate)
        
        # Pool features to fixed size
        pooled_features = self._pool_features(features)
        
        # Gender classification
        gender_logits = self.classifier(pooled_features)
        
        outputs = {
            'gender_logits': gender_logits,
            'features': pooled_features
        }
        
        # Dog identity classification (if enabled)
        if self.use_dog_identity and hasattr(self, 'dog_classifier'):
            dog_logits = self.dog_classifier(pooled_features)
            outputs['dog_logits'] = dog_logits
        
        # Contrastive embeddings (if enabled)
        if self.use_contrastive and hasattr(self, 'projection_head'):
            contrastive_embeddings = self.projection_head(pooled_features)
            outputs['contrastive_embeddings'] = contrastive_embeddings
        
        if return_embeddings:
            outputs['raw_features'] = features
            outputs['pooled_features'] = pooled_features
        
        return outputs
    
    def compute_contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                                dog_ids: Optional[torch.Tensor] = None, temperature: float = 0.07):
        """
        Compute contrastive loss for gender classification.
        
        Args:
            embeddings: L2-normalized embeddings [batch_size, embed_dim]
            labels: Gender labels [batch_size]
            dog_ids: Dog identity labels [batch_size] (optional)
            temperature: Temperature for contrastive loss
        
        Returns:
            Contrastive loss value
        """
        batch_size = embeddings.shape[0]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # Create positive and negative masks
        labels = labels.view(-1, 1)
        same_gender_mask = torch.eq(labels, labels.T).float().to(self.device)
        
        # If dog_ids provided, ensure we don't use same dog as negative
        if dog_ids is not None:
            dog_ids = dog_ids.view(-1, 1)
            same_dog_mask = torch.eq(dog_ids, dog_ids.T).float().to(self.device)
            # Positive pairs: same gender, different dogs
            positive_mask = same_gender_mask * (1 - same_dog_mask)
        else:
            # Positive pairs: same gender, different samples
            positive_mask = same_gender_mask * (1 - torch.eye(batch_size).to(self.device))
        
        # Negative pairs: different gender
        negative_mask = 1 - same_gender_mask
        
        # Compute loss
        # Numerator: exp(similarity) for positive pairs
        pos_exp = torch.exp(similarity_matrix) * positive_mask
        
        # Denominator: exp(similarity) for all pairs except self
        all_exp = torch.exp(similarity_matrix) * (1 - torch.eye(batch_size).to(self.device))
        
        # Avoid division by zero
        pos_sum = pos_exp.sum(dim=1)
        all_sum = all_exp.sum(dim=1)
        
        # Only compute loss for samples that have positive pairs
        valid_samples = (pos_sum > 0).float()
        
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Contrastive loss: -log(sum(pos) / sum(all))
        loss = -torch.log(pos_sum / (all_sum + 1e-8)) * valid_samples
        loss = loss.sum() / (valid_samples.sum() + 1e-8)
        
        return loss
    
    def compute_multi_task_loss(self, outputs: Dict, batch: Dict, 
                               gender_weight: float = 1.0, dog_weight: float = 0.5, 
                               contrastive_weight: float = 0.3):
        """
        Compute multi-task loss combining gender, dog identity, and contrastive losses.
        
        Args:
            outputs: Model outputs dictionary
            batch: Input batch dictionary
            gender_weight: Weight for gender classification loss
            dog_weight: Weight for dog identity classification loss
            contrastive_weight: Weight for contrastive loss
        
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # Gender classification loss
        gender_criterion = nn.CrossEntropyLoss()
        gender_loss = gender_criterion(outputs['gender_logits'], batch['labels'])
        losses['gender_loss'] = gender_loss
        
        total_loss = gender_weight * gender_loss
        
        # Dog identity classification loss (if enabled)
        if 'dog_logits' in outputs and 'dog_ids' in batch:
            dog_criterion = nn.CrossEntropyLoss()
            dog_loss = dog_criterion(outputs['dog_logits'], batch['dog_ids'])
            losses['dog_loss'] = dog_loss
            total_loss += dog_weight * dog_loss
        
        # Contrastive loss (if enabled)
        if 'contrastive_embeddings' in outputs:
            contrastive_loss = self.compute_contrastive_loss(
                outputs['contrastive_embeddings'], 
                batch['labels'],
                batch.get('dog_ids')
            )
            losses['contrastive_loss'] = contrastive_loss
            total_loss += contrastive_weight * contrastive_loss
        
        losses['total_loss'] = total_loss
        return losses


class L2Norm(nn.Module):
    """L2 normalization layer."""
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


def create_dog_aware_wav2vec2_model(num_labels=2, num_dogs=None, device=None, 
                                   model_type="dog_aware", **kwargs):
    """
    Factory function to create dog-aware Wav2Vec2 models.
    
    Args:
        num_labels: Number of gender classes (2)
        num_dogs: Number of individual dogs (for multi-task learning)
        device: torch.device or None (auto-detect)
        model_type: "dog_aware", "contrastive", or "multi_task"
        **kwargs: Additional arguments
    
    Returns:
        DogAwareWav2Vec2Model instance
    """
    if model_type == "dog_aware":
        return DogAwareWav2Vec2Model(
            num_labels=num_labels, 
            num_dogs=num_dogs,
            device=device, 
            use_contrastive=False, 
            use_dog_identity=False,
            **kwargs
        )
    elif model_type == "contrastive":
        return DogAwareWav2Vec2Model(
            num_labels=num_labels, 
            num_dogs=num_dogs,
            device=device, 
            use_contrastive=True, 
            use_dog_identity=False,
            **kwargs
        )
    elif model_type == "multi_task":
        return DogAwareWav2Vec2Model(
            num_labels=num_labels, 
            num_dogs=num_dogs,
            device=device, 
            use_contrastive=True, 
            use_dog_identity=True,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    """Test the dog-aware model."""
    
    print("=== Testing Dog-Aware Wav2Vec2 Model ===\n")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_dog_aware_wav2vec2_model(
        num_labels=2, 
        num_dogs=20,  # Assume 20 different dogs
        device=device,
        model_type="multi_task"
    )
    
    # Load backbone
    model.load_backbone({'model_name': 'facebook/wav2vec2-base'})
    
    # Test with dummy batch
    batch_size = 4
    sequence_length = 16000  # 1 second of audio
    
    dummy_batch = {
        'audio': torch.randn(batch_size, sequence_length).to(device),
        'labels': torch.randint(0, 2, (batch_size,)).to(device),  # Gender labels
        'dog_ids': torch.randint(0, 20, (batch_size,)).to(device),  # Dog identity labels
        'sampling_rate': 16000
    }
    
    # Forward pass
    print("1. Testing forward pass...")
    outputs = model.forward(dummy_batch, return_embeddings=True)
    
    print(f"Gender logits shape: {outputs['gender_logits'].shape}")
    print(f"Dog logits shape: {outputs['dog_logits'].shape}")
    print(f"Contrastive embeddings shape: {outputs['contrastive_embeddings'].shape}")
    print(f"Pooled features shape: {outputs['pooled_features'].shape}")
    
    # Test multi-task loss
    print("\n2. Testing multi-task loss...")
    losses = model.compute_multi_task_loss(outputs, dummy_batch)
    
    print(f"Gender loss: {losses['gender_loss'].item():.4f}")
    print(f"Dog loss: {losses['dog_loss'].item():.4f}")
    print(f"Contrastive loss: {losses['contrastive_loss'].item():.4f}")
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    
    print("\n=== All tests passed! ===")
