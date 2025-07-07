import os
import numpy as np
import torch
from typing import Dict, Any, Optional
from datasets import load_dataset
from transformers import ASTFeatureExtractor

try:
    from .simple_dataset_interface import SimpleBarkopediaDataset, SimpleDatasetConfig
except ImportError:
    # If running as main, use absolute import
    from simple_dataset_interface import SimpleBarkopediaDataset, SimpleDatasetConfig

class SimpleBarkopediaAgeDataset(SimpleBarkopediaDataset):
    """
    Simple implementation for Barkopedia Dog Age Group Classification Dataset.
    
    Age groups:
    - 0: adolescent (18 months - 3 years)
    - 1: adult (3-8 years)
    - 2: juvenile (6-18 months) 
    - 3: puppy (0-6 months)
    - 4: senior (8+ years)
    """
    
    def __init__(self, config: SimpleDatasetConfig, split: str = "train"):
        super().__init__(config, split)
        
        # Age group mapping
        self.id_to_label = {
            0: "adolescent",
            1: "adult", 
            2: "juvenile",
            3: "puppy",
            4: "senior"
        }
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        
        # Set up feature extractor
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        
        self.hf_dataset_name = "ArlingtonCL2/Barkopedia_DOG_AGE_GROUP_CLASSIFICATION_DATASET"
    
    def load_data(self) -> None:
        """Load the Barkopedia age group dataset."""
        print(f"Loading {self.split} data from HuggingFace Hub...")
        
        try:
            # Load from HuggingFace Hub
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
                raw_data = ds[available_splits[0]]
            elif self.split == "test" and len(available_splits) > 1:
                raw_data = ds[available_splits[1]]
            else:
                # Fallback to first available split
                raw_data = ds[available_splits[0]]
                print(f"Warning: {self.split} split not found, using {available_splits[0]}")
            
            print(f"Processing {len(raw_data)} samples...")
            
            # Process each sample
            self.data = []
            self.labels = []
            
            for i, sample in enumerate(raw_data):
                processed_sample = self._process_sample(sample)
                self.data.append(processed_sample)
                self.labels.append(processed_sample['labels'])
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(raw_data)} samples")
            
            print(f"Loaded {len(self.data)} samples for {self.split} split")
            print(f"Label distribution: {self.get_class_distribution()}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    
    def _process_sample(self, sample: Any) -> Dict[str, Any]:
        """Process a single sample into the format expected by models."""
        # Extract audio data
        if isinstance(sample["audio"], dict):
            audio = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
        else:
            raise ValueError("Unexpected audio format in sample")
        
        # Resample if needed
        if sampling_rate != self.config.sampling_rate:
            audio = self.resample_audio(audio, sampling_rate, self.config.sampling_rate)
        
        # Apply cleaning if enabled
        audio = self.clean_audio(audio, self.config.sampling_rate)
        
        # Extract features using AST feature extractor
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=self.config.sampling_rate, 
            return_tensors="pt"
        )
        input_values = inputs.input_values.squeeze(0)  # Remove batch dimension
        
        # Get label information
        label_id = sample["label"]
        label_name = self.id_to_label[label_id]
        
        return {
            "input_values": input_values,
            "labels": label_id,
            "label_name": label_name,
            "sampling_rate": self.config.sampling_rate,
            "audio": audio  # Keep original audio if needed
        }


def create_simple_barkopedia_dataset(
    cache_dir: str = "./barkopedia_dataset",
    hf_token: Optional[str] = None,
    apply_cleaning: bool = False,
    sampling_rate: int = 16000,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None
) -> Dict[str, SimpleBarkopediaAgeDataset]:
    """
    Create simple Barkopedia age group dataset splits.
    
    Args:
        cache_dir: Directory to cache the dataset
        hf_token: HuggingFace token for private datasets
        apply_cleaning: Whether to apply audio cleaning
        sampling_rate: Target sampling rate for audio
        max_duration: Maximum duration in seconds
        min_duration: Minimum duration in seconds
    
    Returns:
        Dictionary with 'train' and 'test' dataset instances
    """
    
    # Create configuration
    config = SimpleDatasetConfig(
        dataset_name="barkopedia_age_group",
        cache_dir=cache_dir,
        hf_token=hf_token,
        sampling_rate=sampling_rate,
        apply_cleaning=apply_cleaning,
        max_duration=max_duration,
        min_duration=min_duration
    )
    
    # Create train and test datasets
    train_dataset = SimpleBarkopediaAgeDataset(config, split="train")
    train_dataset.load_data()
    
    test_dataset = SimpleBarkopediaAgeDataset(config, split="test")
    test_dataset.load_data()
    
    return {
        "train": train_dataset,
        "test": test_dataset
    }


if __name__ == "__main__":
    """Test the simple Barkopedia age group dataset implementation."""
    
    print("=== Testing Simple Barkopedia Age Group Dataset ===\n")
    
    # Load HuggingFace token if available
    hf_token = None
    try:
        hf_token = os.environ.get("HF_TOKEN")
    except:
        pass
    
    # Create dataset with cleaning enabled
    print("1. Creating dataset splits with cleaning enabled...")
    splits = create_simple_barkopedia_dataset(
        hf_token=hf_token,
        apply_cleaning=True,
        max_duration=5.0  # Limit to 5 seconds max
    )
    
    train_dataset = splits["train"]
    test_dataset = splits["test"]
    
    print(f"   ✓ Train dataset: {len(train_dataset)} samples")
    print(f"   ✓ Test dataset: {len(test_dataset)} samples")
    
    # Test sample access
    print("\n2. Testing sample access...")
    train_sample = train_dataset[0]
    
    print(f"   ✓ Sample keys: {list(train_sample.keys())}")
    print(f"   ✓ Input values shape: {train_sample['input_values'].shape}")
    print(f"   ✓ Label: {train_sample['labels']} ({train_sample['label_name']})")
    print(f"   ✓ Sampling rate: {train_sample['sampling_rate']}")
    print(f"   ✓ Audio shape: {train_sample['audio'].shape}")
    
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
