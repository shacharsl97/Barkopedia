from abc import ABC, abstractmethod

class BackboneModel(ABC):
    """
    Abstract base class for all Barkopedia backbone models.
    Each backbone must implement these methods for feature extraction and inference.
    """
    def __init__(self):
        self.model = None  # The main model (e.g., AST, BEATS, Wav2Vec2)
        self.feature_extractor = None  # The feature extractor (e.g., ASTFeatureExtractor)
        self.device = None  # torch.device or str
        self.model_path = None  # Path to model weights
        self.backbone_name = None  # Name of the backbone (e.g., 'ast', 'beats', 'wav2vec2')

    @abstractmethod
    def load(self, model_path: str):
        """Load model weights and backbone."""
        pass

    @abstractmethod
    def forward(self, audio, sampling_rate: int):
        """Forward pass: extract features from raw audio using the backbone."""
        pass


class ClassificationModel(ABC):
    """
    Abstract base class for Barkopedia classification models.
    These models use a backbone for feature extraction and perform classification.
    """
    def __init__(self):
        self.backbone = None  # Instance of BackboneModel
        self.classifier = None  # Classification head (e.g., linear layer, MLP)
        self.device = None
        self.model_path = None

    @abstractmethod
    def load(self, model_path: str):
        """Load the backbone and classifier weights."""
        pass

    @abstractmethod
    def load_backbone(self, backbone_args: dict):
        """Instantiate and load the backbone model with the given arguments (e.g., model name, config, etc)."""
        pass

    @abstractmethod
    def forward(self, audio, sampling_rate: int):
        """Forward pass through the full model (backbone + classifier)."""
        pass

    @abstractmethod
    def predict(self, audio, sampling_rate: int):
        """Run inference and return class prediction(s) for the input audio."""
        pass
        pass
