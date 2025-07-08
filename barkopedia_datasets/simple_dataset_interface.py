from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

@dataclass
class SimpleDatasetConfig:
    """Simple configuration for Barkopedia datasets."""
    dataset_name: str
    cache_dir: str = "./cache"
    hf_token: Optional[str] = None
    sampling_rate: int = 16000
    apply_cleaning: bool = False
    max_duration: Optional[float] = None  # seconds
    min_duration: Optional[float] = None  # seconds
    # Segmentation options
    enable_segmentation: bool = False  # Enable automatic segmentation
    segment_duration: float = 2.0  # Target segment duration in seconds (0.3-5.0)
    segment_overlap: float = 0.1   # Overlap between segments in seconds
    min_segment_duration: float = 0.3  # Minimum segment duration in seconds
    max_segment_duration: float = 5.0  # Maximum segment duration in seconds
    energy_threshold: float = 0.01  # Energy threshold for silence detection (0.001-0.1)
    silence_min_duration: float = 0.1  # Minimum silence duration to split on (seconds)

class SimpleBarkopediaDataset(Dataset, ABC):
    """
    Simplified base class for Barkopedia datasets.
    Just holds data and provides PyTorch Dataset interface.
    """
    
    def __init__(self, config: SimpleDatasetConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.data = []
        self.labels = []
        self.id_to_label = {}
        self.label_to_id = {}
    
    @abstractmethod
    def load_data(self) -> None:
        """Load the raw dataset from source."""
        pass
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""
        return self.data[idx]
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, **kwargs) -> DataLoader:
        """Create a PyTorch DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            **kwargs
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching samples with padding."""
        from torch.nn.utils.rnn import pad_sequence
        
        # Convert input_values to tensors and pad to same length
        input_values_list = []
        for item in batch:
            if isinstance(item['input_values'], torch.Tensor):
                input_values_list.append(item['input_values'])
            else:
                input_values_list.append(torch.tensor(item['input_values'], dtype=torch.float32))
        
        # Pad sequences to the same length
        input_values = pad_sequence(input_values_list, batch_first=True, padding_value=0.0)
        
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        
        return {
            'input_values': input_values,
            'labels': labels
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {}
        for label in self.labels:
            label_name = self.id_to_label.get(label, str(label))
            distribution[label_name] = distribution.get(label_name, 0) + 1
        return distribution
    
    def clean_audio(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Apply basic audio cleaning if enabled."""
        if not self.config.apply_cleaning:
            return audio
            
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Trim silence from beginning and end (simple version)
        try:
            import librosa
            audio, _ = librosa.effects.trim(audio, top_db=20)
        except ImportError:
            # Simple fallback
            threshold = 0.01
            start = 0
            end = len(audio)
            
            for i, sample in enumerate(audio):
                if abs(sample) > threshold:
                    start = i
                    break
            
            for i in range(len(audio) - 1, -1, -1):
                if abs(audio[i]) > threshold:
                    end = i + 1
                    break
                    
            audio = audio[start:end]
        
        # Apply duration constraints
        if self.config.max_duration:
            max_samples = int(self.config.max_duration * sampling_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
        
        if self.config.min_duration:
            min_samples = int(self.config.min_duration * sampling_rate)
            if len(audio) < min_samples:
                # Pad with zeros
                padding = min_samples - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio
    
    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple audio resampling."""
        if orig_sr == target_sr:
            return audio
        
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)
