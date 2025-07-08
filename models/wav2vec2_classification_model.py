import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2Model

try:
    from .model_interface import BackboneModel, ClassificationModel
except ImportError:
    # If running as main, use absolute import
    from model_interface import BackboneModel, ClassificationModel

class Wav2Vec2Backbone(BackboneModel):
    def __init__(self, model_name="facebook/wav2vec2-base", device=None):
        super().__init__()
        self.model_name = model_name
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone_name = "wav2vec2"

    def load(self, model_path=None):
        """Load the Wav2Vec2 model and feature extractor."""
        model_path = model_path or self.model_name
        
        # Load the base model (without classification head)
        self.model = Wav2Vec2Model.from_pretrained(model_path).to(self.device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model_path = model_path
        
        # Freeze the backbone for feature extraction
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, audio, sampling_rate: int):
        """Extract features from audio using Wav2Vec2 backbone."""
        # Process audio with feature extractor
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=sampling_rate, 
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            # Forward pass through Wav2Vec2 model
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get the last hidden state
            features = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]
            
        return features

class Wav2Vec2ClassificationModel(ClassificationModel):
    def __init__(self, num_labels=2, device=None, pooling_mode="mean"):
        super().__init__()
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_labels = num_labels
        self.pooling_mode = pooling_mode  # "mean", "max", "cls"
        self.feature_extractor = None  # Will be set when loading
        
        # Wav2Vec2 base model has 768 hidden dimensions
        self.hidden_size = 768
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_labels)
        ).to(self.device)

    def load_backbone(self, backbone_args: dict):
        """Load the Wav2Vec2 backbone model."""
        self.backbone = Wav2Vec2Backbone(**backbone_args, device=self.device)
        self.backbone.load()
        self.feature_extractor = self.backbone.feature_extractor

    def load(self, model_path: str):
        """Load pre-trained Wav2Vec2 model (currently uses default backbone)."""
        self.backbone = Wav2Vec2Backbone(device=self.device)
        self.backbone.load(model_path)
        self.feature_extractor = self.backbone.feature_extractor
        self.model_path = model_path

    def _pool_features(self, features):
        """Pool sequence features to fixed-size representation."""
        if self.pooling_mode == "mean":
            # Mean pooling over sequence dimension
            return torch.mean(features, dim=1)  # [batch_size, hidden_size]
        elif self.pooling_mode == "max":
            # Max pooling over sequence dimension
            return torch.max(features, dim=1)[0]  # [batch_size, hidden_size]
        elif self.pooling_mode == "cls":
            # Use first token (like CLS token)
            return features[:, 0, :]  # [batch_size, hidden_size]
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

    def forward(self, audio, sampling_rate: int):
        """Forward pass through Wav2Vec2 backbone + classifier."""
        # Extract features using backbone
        features = self.backbone.forward(audio, sampling_rate)
        
        # Pool features to fixed size
        pooled_features = self._pool_features(features)
        
        # Forward through classifier
        logits = self.classifier(pooled_features)
        
        return logits

    def predict(self, audio, sampling_rate: int):
        """Run inference and return class prediction(s) for the input audio."""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            logits = self.forward(audio, sampling_rate)
            predictions = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1)
            return predicted_class.cpu().numpy(), predictions.cpu().numpy()

    def eval(self):
        """Set model to evaluation mode."""
        if self.backbone and hasattr(self.backbone, 'model'):
            self.backbone.model.eval()
        if self.classifier:
            self.classifier.eval()

    def save(self, save_path: str):
        """Save the classification model."""
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'num_labels': self.num_labels,
            'pooling_mode': self.pooling_mode,
            'hidden_size': self.hidden_size,
            'backbone_model_name': self.backbone.model_name if self.backbone else None,
        }, save_path)

    def load_classifier(self, save_path: str):
        """Load saved classification head."""
        checkpoint = torch.load(save_path, map_location=self.device)
        
        # Restore configuration
        self.num_labels = checkpoint['num_labels']
        self.pooling_mode = checkpoint['pooling_mode']
        self.hidden_size = checkpoint['hidden_size']
        
        # Recreate classifier with saved config
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, self.num_labels)
        ).to(self.device)
        
        # Load classifier weights
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])


# Alternative implementation using the pre-trained sequence classification model
class Wav2Vec2PretrainedClassificationModel(ClassificationModel):
    """
    Alternative implementation using Wav2Vec2ForSequenceClassification directly.
    This is useful when you want to fine-tune the entire model end-to-end.
    """
    def __init__(self, num_labels=2, device=None):
        super().__init__()
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_labels = num_labels
        self.feature_extractor = None
        self.model = None

    def load_backbone(self, backbone_args: dict):
        """Load pre-trained Wav2Vec2 model with classification head."""
        model_name = backbone_args.get('model_name', 'facebook/wav2vec2-base')
        
        # Load the feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # Load the model with classification head
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True  # Allow different number of labels
        ).to(self.device)

    def load(self, model_path: str):
        """Load saved model."""
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model_path = model_path

    def forward(self, audio, sampling_rate: int):
        """Forward pass through the full model."""
        # Process audio with feature extractor
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=sampling_rate, 
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Forward pass through model
        outputs = self.model(**inputs)
        return outputs.logits

    def predict(self, audio, sampling_rate: int):
        """Run inference and return class prediction(s) for the input audio."""
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():
            logits = self.forward(audio, sampling_rate)
            predictions = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1)
            return predicted_class.cpu().numpy(), predictions.cpu().numpy()

    def save(self, save_path: str):
        """Save the entire model."""
        self.model.save_pretrained(save_path)
        self.feature_extractor.save_pretrained(save_path)


# Factory function for easy instantiation
def create_wav2vec2_model(num_labels=2, device=None, model_type="custom", **kwargs):
    """
    Factory function to create Wav2Vec2 classification models.
    
    Args:
        num_labels: Number of output classes
        device: torch.device or None (auto-detect)
        model_type: "custom" or "pretrained"
        **kwargs: Additional arguments passed to the model
    
    Returns:
        Wav2Vec2ClassificationModel or Wav2Vec2PretrainedClassificationModel
    """
    if model_type == "custom":
        return Wav2Vec2ClassificationModel(num_labels=num_labels, device=device, **kwargs)
    elif model_type == "pretrained":
        return Wav2Vec2PretrainedClassificationModel(num_labels=num_labels, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'custom' or 'pretrained'")


if __name__ == "__main__":
    # Test the model
    import numpy as np
    
    # Create model
    model = create_wav2vec2_model(num_labels=2, model_type="custom")
    
    # Load backbone
    model.load_backbone({'model_name': 'facebook/wav2vec2-base'})
    
    # Test with dummy audio
    dummy_audio = np.random.randn(16000)  # 1 second of audio at 16kHz
    
    # Forward pass
    logits = model.forward(dummy_audio, 16000)
    print(f"Output shape: {logits.shape}")
    print(f"Logits: {logits}")
    
    # Test predictions
    predictions = torch.softmax(logits, dim=-1)
    print(f"Predictions: {predictions}")
