import os
import numpy as np
import torch
from typing import Dict, Any, Optional, List
from datasets import load_dataset, load_from_disk
from transformers import ASTFeatureExtractor

from .dataset_interface import BarkopediaDataset, DatasetConfig, AudioDatasetMixin

class BarkopediaGenderDataset(BarkopediaDataset, AudioDatasetMixin):
    """
    Concrete implementation for Barkopedia Dog Gender/Sex Classification Dataset.
    
    This dataset contains dog barks labeled with gender:
    - female (0)
    - male (1)
    """
    
    def __init__(self, config: DatasetConfig, split: str = "train"):
        super().__init__(config)
        self.split = split
        self.hf_dataset_name = "ArlingtonCL2/Barkopedia_Dog_Sex_Classification_Dataset"
        
        # Gender mapping
        self.id_to_label = {
            0: "female",
            1: "male"
        }
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        
        # Set up feature extractor
        if self.feature_extractor is None:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
    
    def load_data(self) -> None:
        """Load the Barkopedia gender classification dataset."""
        cache_path = os.path.join(self.config.cache_dir, f"{self.split}_preprocessed")
        
        # Try to load preprocessed data first
        if os.path.exists(cache_path):
            print(f"Loading preprocessed {self.split} data from {cache_path}")
            self.load_preprocessed(cache_path)
            return
        
        # Load from HuggingFace Hub
        print(f"Loading {self.split} data from HuggingFace Hub...")
        try:
            ds = load_dataset(
                self.hf_dataset_name,
                token=self.config.hf_token,
                cache_dir=self.config.cache_dir
            )
            
            # Get available splits
            available_splits = list(ds.keys())
            print(f"Available splits: {available_splits}")
            
            # Map split names
            if self.split == "train" and len(available_splits) > 0:
                self.data = ds[available_splits[0]]
            elif self.split == "test" and len(available_splits) > 1:
                self.data = ds[available_splits[1]]
            elif self.split == "validation" and len(available_splits) > 2:
                self.data = ds[available_splits[2]]
            else:
                # Fallback to first available split
                self.data = ds[available_splits[0]]
                print(f"Warning: {self.split} split not found, using {available_splits[0]}")
            
            # Extract labels
            self.labels = [sample['label'] for sample in self.data]
            
            # Update config with dataset info
            if self.config.num_labels is None:
                self.config.num_labels = len(self.id_to_label)
            if self.config.label_names is None:
                self.config.label_names = list(self.id_to_label.values())
            
            print(f"Loaded {len(self.data)} samples for {self.split} split")
            print(f"Label distribution: {self.get_class_distribution()}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    
    def get_splits(self) -> Dict[str, 'BarkopediaGenderDataset']:
        """Return train and test splits as separate dataset instances."""
        splits = {}
        
        # Create train split
        train_config = DatasetConfig(
            dataset_name=self.config.dataset_name,
            cache_dir=self.config.cache_dir,
            hf_token=self.config.hf_token,
            num_labels=self.config.num_labels,
            label_names=self.config.label_names,
            sampling_rate=self.config.sampling_rate,
            augmentation_enabled=self.config.augmentation_enabled,
            augmentation_params=self.config.augmentation_params
        )
        
        train_dataset = BarkopediaGenderDataset(train_config, split="train")
        train_dataset.load_data()
        splits["train"] = train_dataset
        
        # Create test split
        test_config = DatasetConfig(
            dataset_name=self.config.dataset_name,
            cache_dir=self.config.cache_dir,
            hf_token=self.config.hf_token,
            num_labels=self.config.num_labels,
            label_names=self.config.label_names,
            sampling_rate=self.config.sampling_rate,
            augmentation_enabled=False,  # Usually no augmentation for test
            augmentation_params=None
        )
        
        test_dataset = BarkopediaGenderDataset(test_config, split="test")
        test_dataset.load_data()
        splits["test"] = test_dataset
        
        return splits
    
    def preprocess_audio(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Preprocess raw audio data."""
        # Resample if needed
        if sampling_rate != self.config.sampling_rate:
            audio = self.resample_audio(audio, sampling_rate, self.config.sampling_rate)
        
        # Normalize audio
        if self.config.normalize_audio:
            audio = self.normalize_audio(audio)
        
        # Trim silence (optional)
        audio = self.trim_silence(audio)
        
        # Apply duration constraints
        if self.config.max_duration:
            max_samples = int(self.config.max_duration * self.config.sampling_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
        
        if self.config.min_duration:
            min_samples = int(self.config.min_duration * self.config.sampling_rate)
            if len(audio) < min_samples:
                audio = self.pad_or_truncate(audio, min_samples)
        
        return audio
    
    def extract_features(self, audio: np.ndarray, sampling_rate: int) -> torch.Tensor:
        """Extract features using the AST feature extractor."""
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        )
        return inputs.input_values.squeeze(0)  # Remove batch dimension
    
    def _process_sample(self, sample: Any) -> Dict[str, Any]:
        """Process a single sample and return standardized format."""
        # Extract audio data
        if isinstance(sample["audio"], dict):
            audio = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            audio_path = sample["audio"].get("path", "")
        else:
            raise ValueError("Unexpected audio format in sample")
        
        # Preprocess audio
        audio = self.preprocess_audio(audio, sampling_rate)
        
        # Apply augmentation if enabled
        augmented_samples = self.apply_augmentation(audio, self.config.sampling_rate)
        
        # For now, just use the first sample (original or first augmentation)
        audio, aug_type = augmented_samples[0]
        
        # Extract features
        input_values = self.extract_features(audio, self.config.sampling_rate)
        
        # Get label information
        label_id = sample["label"]
        label_name = self.id_to_label[label_id]
        
        return {
            "audio": audio,
            "input_values": input_values,
            "labels": label_id,
            "label_name": label_name,
            "sampling_rate": self.config.sampling_rate,
            "metadata": {
                "audio_path": audio_path,
                "augmentation_type": aug_type,
                "original_sampling_rate": sampling_rate
            }
        }


def create_barkopedia_gender_dataset(
    cache_dir: str = "./barkopedia_dataset",
    hf_token: Optional[str] = None,
    augmentation: bool = False,
    sampling_rate: int = 16000
) -> Dict[str, BarkopediaGenderDataset]:
    """
    Convenience function to create Barkopedia gender classification dataset splits.
    
    Args:
        cache_dir: Directory to cache the dataset
        hf_token: HuggingFace token for private datasets
        augmentation: Whether to enable data augmentation for training
        sampling_rate: Target sampling rate for audio
    
    Returns:
        Dictionary with 'train' and 'test' dataset instances
    """
    
    # Set up augmentation parameters
    aug_params = None
    if augmentation:
        aug_params = {
            "add_noise": True,
            "noise_factor": 0.005,
            "pitch_shift": True,
            "pitch_shift_steps": 2,
            "time_stretch": True,
            "time_stretch_rate": 0.8
        }
    
    # Create configuration
    config = DatasetConfig(
        dataset_name="barkopedia_gender",
        cache_dir=cache_dir,
        hf_token=hf_token,
        sampling_rate=sampling_rate,
        normalize_audio=True,
        augmentation_enabled=augmentation,
        augmentation_params=aug_params
    )
    
    # Create dataset instance and get splits
    dataset = BarkopediaGenderDataset(config)
    splits = dataset.get_splits()
    
    return splits


if __name__ == "__main__":
    """Test the Barkopedia gender classification dataset implementation."""
    
    print("=== Testing Barkopedia Gender Classification Dataset ===\n")
    
    # Load HuggingFace token if available
    hf_token = None
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from TASK1_age_group.hf_token import HF_TOKEN
        hf_token = HF_TOKEN
    except:
        hf_token = os.environ.get("HF_TOKEN")
    
    # Create dataset
    print("1. Creating dataset splits...")
    splits = create_barkopedia_gender_dataset(
        hf_token=hf_token,
        augmentation=False  # Start without augmentation for testing
    )
    
    train_dataset = splits["train"]
    test_dataset = splits["test"]
    
    print(f"   ✓ Train dataset: {len(train_dataset)} samples")
    print(f"   ✓ Test dataset: {len(test_dataset)} samples")
    
    # Test sample processing
    print("\n2. Testing sample processing...")
    train_sample = train_dataset[0]
    
    print(f"   ✓ Sample keys: {train_sample.keys()}")
    print(f"   ✓ Audio shape: {train_sample['audio'].shape}")
    print(f"   ✓ Input values shape: {train_sample['input_values'].shape}")
    print(f"   ✓ Label: {train_sample['labels']} ({train_sample['label_name']})")
    print(f"   ✓ Sampling rate: {train_sample['sampling_rate']}")
    
    # Test class distribution
    print("\n3. Class distribution:")
    train_dist = train_dataset.get_class_distribution()
    for label, count in train_dist.items():
        print(f"   {label}: {count} samples")
    
    # Test DataLoader
    print("\n4. Testing DataLoader...")
    dataloader = train_dataset.get_dataloader(batch_size=4, shuffle=False)
    batch = next(iter(dataloader))
    
    print(f"   ✓ Batch input_values shape: {batch['input_values'].shape}")
    print(f"   ✓ Batch labels shape: {batch['labels'].shape}")
    print(f"   ✓ Labels in batch: {batch['labels']}")
    
    print("\n=== All tests passed! ===")
