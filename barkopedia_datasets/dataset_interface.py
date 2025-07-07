from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Union
import torch
from torch.utils.data import Dataset
import numpy as np

@dataclass
class DatasetConfig:
    """Configuration class for Barkopedia datasets."""
    dataset_name: str
    cache_dir: str = "./barkopedia_dataset"
    hf_token: Optional[str] = None
    num_labels: Optional[int] = None
    label_names: Optional[List[str]] = None
    sampling_rate: int = 16000
    max_duration: Optional[float] = None  # in seconds
    min_duration: Optional[float] = None  # in seconds
    normalize_audio: bool = True
    augmentation_enabled: bool = False
    augmentation_params: Optional[Dict[str, Any]] = None

class BarkopediaDataset(ABC, Dataset):
    """
    Abstract base class for all Barkopedia datasets.
    
    This interface provides a standardized way to:
    1. Load and preprocess audio datasets
    2. Handle data augmentation
    3. Provide train/validation/test splits
    4. Convert between different audio formats and feature representations
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data = None
        self.labels = None
        self.label_to_id = {}
        self.id_to_label = {}
        self.feature_extractor = None
        
    @abstractmethod
    def load_data(self) -> None:
        """Load the dataset from source (HuggingFace Hub, local files, etc.)."""
        pass
    
    @abstractmethod
    def get_splits(self) -> Dict[str, 'BarkopediaDataset']:
        """Return train, validation, and test splits as separate dataset instances."""
        pass
    
    @abstractmethod
    def preprocess_audio(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Preprocess raw audio data (normalization, resampling, etc.)."""
        pass
    
    @abstractmethod
    def extract_features(self, audio: np.ndarray, sampling_rate: int) -> torch.Tensor:
        """Extract features from audio using the configured feature extractor."""
        pass
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load_data() first.")
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dictionary containing:
            - 'audio': preprocessed audio array
            - 'input_values': feature extractor output
            - 'labels': label ID (int)
            - 'label_name': label name (str)
            - 'sampling_rate': sampling rate
            - 'metadata': additional metadata (optional)
        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load_data() first.")
        
        sample = self.data[idx]
        return self._process_sample(sample)
    
    @abstractmethod
    def _process_sample(self, sample: Any) -> Dict[str, Any]:
        """Process a single sample and return standardized format."""
        pass
    
    def get_label_info(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Return label mapping dictionaries."""
        return self.id_to_label, self.label_to_id
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Return the distribution of classes in the dataset."""
        if self.labels is None:
            raise RuntimeError("Dataset not loaded. Call load_data() first.")
        
        distribution = {}
        for label_id in self.labels:
            label_name = self.id_to_label.get(label_id, str(label_id))
            distribution[label_name] = distribution.get(label_name, 0) + 1
        return distribution
    
    def apply_augmentation(self, audio: np.ndarray, sampling_rate: int) -> List[Tuple[np.ndarray, str]]:
        """
        Apply data augmentation to audio.
        
        Returns:
            List of tuples (augmented_audio, augmentation_type)
        """
        if not self.config.augmentation_enabled or not self.config.augmentation_params:
            return [(audio, "original")]
        
        augmented_samples = [(audio, "original")]
        params = self.config.augmentation_params
        
        try:
            import librosa
            
            if params.get("add_noise", False):
                noise_factor = params.get("noise_factor", 0.005)
                noisy_audio = audio + noise_factor * np.random.randn(*audio.shape)
                augmented_samples.append((noisy_audio, "noise"))
            
            if params.get("pitch_shift", False):
                n_steps = params.get("pitch_shift_steps", 2)
                pitched_audio = librosa.effects.pitch_shift(y=audio, sr=sampling_rate, n_steps=n_steps)
                augmented_samples.append((pitched_audio, "pitch_shift"))
            
            if params.get("time_stretch", False):
                stretch_rate = params.get("time_stretch_rate", 0.8)
                stretched_audio = librosa.effects.time_stretch(audio, rate=stretch_rate)
                augmented_samples.append((stretched_audio, "time_stretch"))
                
        except ImportError:
            print("Warning: librosa not available for augmentation")
        
        return augmented_samples
    
    def save_preprocessed(self, save_path: str) -> None:
        """Save preprocessed dataset to disk for faster loading."""
        try:
            from datasets import Dataset as HFDataset
            
            if self.data is None:
                raise RuntimeError("Dataset not loaded. Call load_data() first.")
            
            # Convert to HuggingFace dataset format for efficient saving
            processed_data = []
            for i in range(len(self)):
                sample = self[i]
                processed_data.append(sample)
            
            hf_dataset = HFDataset.from_list(processed_data)
            hf_dataset.save_to_disk(save_path)
            print(f"Preprocessed dataset saved to {save_path}")
            
        except ImportError:
            print("Warning: datasets library not available for saving")
    
    def load_preprocessed(self, load_path: str) -> None:
        """Load preprocessed dataset from disk."""
        try:
            from datasets import load_from_disk
            
            self.data = load_from_disk(load_path)
            self.labels = [sample['labels'] for sample in self.data]
            print(f"Preprocessed dataset loaded from {load_path}")
            
        except ImportError:
            print("Warning: datasets library not available for loading")
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4) -> torch.utils.data.DataLoader:
        """Create a PyTorch DataLoader for this dataset."""
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for DataLoader."""
        collated = {}
        
        # Handle input_values (pad sequences if needed)
        input_values = [item['input_values'] for item in batch]
        if isinstance(input_values[0], torch.Tensor):
            collated['input_values'] = torch.stack(input_values)
        else:
            # Pad sequences to same length
            max_len = max(len(iv) for iv in input_values)
            padded = []
            for iv in input_values:
                if len(iv) < max_len:
                    pad_len = max_len - len(iv)
                    iv = torch.cat([torch.tensor(iv), torch.zeros(pad_len)])
                padded.append(iv)
            collated['input_values'] = torch.stack(padded)
        
        # Handle labels
        collated['labels'] = torch.tensor([item['labels'] for item in batch])
        
        # Handle other fields
        for key in ['label_name', 'sampling_rate']:
            if key in batch[0]:
                collated[key] = [item[key] for item in batch]
        
        return collated


class AudioDatasetMixin:
    """Mixin class providing common audio processing utilities."""
    
    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
        return audio
    
    @staticmethod
    def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove silence from beginning and end of audio."""
        try:
            import librosa
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            return audio_trimmed
        except ImportError:
            # Fallback: simple threshold-based trimming
            start_idx = 0
            end_idx = len(audio)
            
            for i, sample in enumerate(audio):
                if abs(sample) > threshold:
                    start_idx = i
                    break
            
            for i in range(len(audio) - 1, -1, -1):
                if abs(audio[i]) > threshold:
                    end_idx = i + 1
                    break
            
            return audio[start_idx:end_idx]
    
    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sampling rate."""
        if orig_sr == target_sr:
            return audio
        
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple resampling fallback (not ideal but functional)
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)
    
    @staticmethod
    def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate audio to target length."""
        if len(audio) == target_length:
            return audio
        elif len(audio) < target_length:
            # Pad with zeros
            pad_length = target_length - len(audio)
            return np.pad(audio, (0, pad_length), mode='constant')
        else:
            # Truncate
            return audio[:target_length]
