import torch
import torch.nn as nn
from transformers import ASTFeatureExtractor, ASTForAudioClassification

try:
    from .model_interface import BackboneModel, ClassificationModel
except ImportError:
    # If running as main, use absolute import
    from model_interface import BackboneModel, ClassificationModel

class ASTBackbone(BackboneModel):
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", device=None):
        super().__init__()
        self.model_name = model_name
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone_name = "ast"

    def load(self, model_path=None):
        self.model = ASTForAudioClassification.from_pretrained(model_path or self.model_name).to(self.device)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_path or self.model_name)
        self.model_path = model_path or self.model_name

    def forward(self, audio, sampling_rate: int):
        inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Use the full model output and extract hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get the last hidden state from the transformer
            features = outputs.hidden_states[-1]  # Shape: [batch_size, sequence_length, hidden_size]
        return features

class ASTClassificationModel(ClassificationModel):
    def __init__(self, num_labels=5, device=None):
        super().__init__()
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_labels = num_labels
        self.feature_extractor = None  # Will be set when loading
        self.classifier = nn.Linear(768, num_labels).to(self.device)  # AST hidden size is 768

    def load_backbone(self, backbone_args: dict):
        self.backbone = ASTBackbone(**backbone_args, device=self.device)
        self.backbone.load()
        self.feature_extractor = self.backbone.feature_extractor

    def load(self, model_path: str):
        self.backbone = ASTBackbone(device=self.device)
        self.backbone.load(model_path)
        # Use the backbone's feature extractor
        self.feature_extractor = self.backbone.feature_extractor
        self.model_path = model_path

    def forward(self, audio, sampling_rate: int):
        """Forward pass through the model."""
        if self.backbone is None:
            raise RuntimeError("Model not loaded. Call load() or load_backbone() first.")
        
        # Extract features using AST backbone
        features = self.backbone.forward(audio, sampling_rate)
        # Pool features (mean pooling over sequence dimension)
        pooled_features = torch.mean(features, dim=1)  # [batch_size, hidden_size]
        # Classify
        logits = self.classifier(pooled_features)
        return logits

    def predict(self, audio, sampling_rate: int):
        """Run inference and return class prediction for the input audio."""
        if self.backbone is None:
            raise RuntimeError("Model not loaded. Call load() or load_backbone() first.")
            
        self.backbone.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            logits = self.forward(audio, sampling_rate)
            pred_id = torch.argmax(logits, dim=1).item()
        return pred_id


if __name__ == "__main__":
    print("Testing ASTClassificationModel loading...")
    import os
    import numpy as np
    
    # Check if local fine-tuned model exists
    local_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "barkopedia_finetuned_model"))
    
    if os.path.exists(local_model_path):
        model_path = local_model_path
        print(f"Using local fine-tuned model: {model_path}")
    else:
        # Fall back to default pretrained model
        model_path = "MIT/ast-finetuned-audioset-10-10-0.4593"
        print(f"Local model not found. Using default pretrained model: {model_path}")
    
    try:
        model = ASTClassificationModel()
        model.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        print("Backbone name:", model.backbone.backbone_name)
        print("Device:", model.device)
        print("Model architecture:", type(model.backbone.model).__name__)
        print("Feature extractor:", type(model.backbone.feature_extractor).__name__)
        print("Classifier head:", type(model.classifier).__name__)
        
        # Test forward pass with dummy audio
        print("\nTesting forward pass with dummy audio...")
        dummy_audio = np.random.randn(16000)  # 1 second of random audio at 16kHz
        sampling_rate = 16000
        
        logits = model.forward(dummy_audio, sampling_rate)
        print(f"Forward pass successful! Output shape: {logits.shape}")
        print(f"Logits: {logits}")
        
        # Test prediction
        print("\nTesting prediction...")
        pred_id = model.predict(dummy_audio, sampling_rate)
        print(f"Predicted class ID: {pred_id}")
        
        print("\nAll tests passed: Model, forward pass, and prediction work correctly.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
