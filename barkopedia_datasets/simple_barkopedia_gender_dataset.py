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

class SimpleBarkopediaGenderDataset(SimpleBarkopediaDataset):
    """
    Simple implementation for Barkopedia Dog Gender/Sex Classification Dataset.
    
    Gender/Sex groups:
    - 0: female
    - 1: male
    """
    
    def __init__(self, config: SimpleDatasetConfig, split: str = "train"):
        super().__init__(config, split)
        
        # Gender mapping (initially - will be updated based on actual data)
        self.id_to_label = {
            0: "female",
            1: "male"
        }
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        
        # Set up feature extractor
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        
        self.hf_dataset_name = "ArlingtonCL2/Barkopedia_Dog_Sex_Classification_Dataset"
    
    def load_data(self) -> None:
        """Load the Barkopedia gender classification dataset."""
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
            
            # Cast audio column to disable automatic decoding - this prevents corruption errors
            from datasets import Audio
            raw_data = raw_data.cast_column("audio", Audio(decode=False))
            
            # Process each sample with robust error handling
            self.data = []
            self.labels = []
            unique_labels = set()
            processed_count = 0
            skipped_count = 0
            
            for i in range(len(raw_data)):
                try:
                    # Get sample - audio will now be a path/bytes, not pre-decoded
                    sample_dict = raw_data[i]
                    
                    # Try to process the sample
                    processed_sample = self._process_sample(sample_dict)
                    self.data.append(processed_sample)
                    self.labels.append(processed_sample['labels'])
                    unique_labels.add(processed_sample['labels'])
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} valid samples (skipped: {skipped_count})...")
                        
                except Exception as e:
                    # Skip corrupted samples
                    skipped_count += 1
                    if skipped_count <= 10:  # Only print first 10 errors to avoid spam
                        print(f"Skipping corrupted sample {i+1}: {str(e)[:100]}...")
                    elif skipped_count == 11:
                        print("... (suppressing further error messages)")
                    continue
            
            # Update label mapping based on discovered labels
            unique_labels = sorted(list(unique_labels))
            print(f"Found unique labels: {unique_labels}")
            
            if len(unique_labels) == 2:
                # Binary classification - assume 0=female, 1=male
                self.id_to_label = {
                    unique_labels[0]: "female", 
                    unique_labels[1]: "male"
                }
            else:
                # Multiple labels - create generic mapping
                self.id_to_label = {label: f"class_{label}" for label in unique_labels}
            
            self.label_to_id = {v: k for k, v in self.id_to_label.items()}
            print(f"Updated label mapping: {self.id_to_label}")
            
            print(f"Loaded {len(self.data)} samples for {self.split} split (skipped: {skipped_count})")
            print(f"Label distribution: {self.get_class_distribution()}")
            
            if len(self.data) == 0:
                raise RuntimeError("No valid samples found in dataset after filtering corrupted files")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    
    def _process_sample(self, sample: Any) -> Dict[str, Any]:
        """Process a single sample into the format expected by models."""
        # Extract audio data - now audio is not pre-decoded due to decode=False
        audio_info = sample["audio"]
        
        if isinstance(audio_info, dict):
            if "array" in audio_info and audio_info["array"] is not None:
                # Audio is already loaded (shouldn't happen with decode=False, but handle it)
                audio = audio_info["array"]
                sampling_rate = audio_info["sampling_rate"]
            elif "path" in audio_info:
                # Audio needs to be loaded from path
                audio_path = audio_info["path"]
                try:
                    import soundfile as sf
                    audio, sampling_rate = sf.read(audio_path)
                except Exception as e:
                    raise ValueError(f"Could not load audio from {audio_path}: {e}")
            elif "bytes" in audio_info:
                # Audio is in bytes format
                try:
                    import soundfile as sf
                    import io
                    audio, sampling_rate = sf.read(io.BytesIO(audio_info["bytes"]))
                except Exception as e:
                    raise ValueError(f"Could not load audio from bytes: {e}")
            else:
                raise ValueError(f"Unknown audio format in sample: {list(audio_info.keys())}")
        else:
            # Direct path or bytes
            try:
                import soundfile as sf
                if isinstance(audio_info, str):
                    # It's a path
                    audio, sampling_rate = sf.read(audio_info)
                else:
                    # It's bytes
                    import io
                    audio, sampling_rate = sf.read(io.BytesIO(audio_info))
            except Exception as e:
                raise ValueError(f"Could not load audio: {e}")
        
        # Convert to numpy array if needed
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        # Ensure correct data type (float32 for PyTorch compatibility)
        audio = audio.astype(np.float32)
        
        # Convert stereo to mono if needed
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)  # Average the channels
        
        # Ensure audio is 1D
        if len(audio.shape) > 1:
            audio = audio.flatten()
        
        # Resample if needed
        if sampling_rate != self.config.sampling_rate:
            audio = self.resample_audio(audio, sampling_rate, self.config.sampling_rate)
        
        # Apply cleaning if enabled
        audio = self.clean_audio(audio, self.config.sampling_rate)
        
        # Final ensure float32 type after all processing
        audio = audio.astype(np.float32)
        
        # IMPORTANT: Return raw audio, not pre-processed features
        # The model will handle feature extraction during training
        
        # Get label information
        label_id = sample["label"]
        label_name = self.id_to_label.get(label_id, f"unknown_{label_id}")
        
        return {
            "audio": audio,  # Raw audio array
            "labels": label_id,
            "label_name": label_name,
            "sampling_rate": self.config.sampling_rate
        }


def create_simple_barkopedia_gender_dataset(
    cache_dir: str = "./barkopedia_dataset",
    hf_token: Optional[str] = None,
    apply_cleaning: bool = False,
    sampling_rate: int = 16000,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None
) -> Dict[str, SimpleBarkopediaGenderDataset]:
    """
    Create simple Barkopedia gender classification dataset splits.
    
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
        dataset_name="barkopedia_gender",
        cache_dir=cache_dir,
        hf_token=hf_token,
        sampling_rate=sampling_rate,
        apply_cleaning=apply_cleaning,
        max_duration=max_duration,
        min_duration=min_duration
    )
    
    # Create train and test datasets
    train_dataset = SimpleBarkopediaGenderDataset(config, split="train")
    train_dataset.load_data()
    
    test_dataset = SimpleBarkopediaGenderDataset(config, split="test")
    test_dataset.load_data()
    
    return {
        "train": train_dataset,
        "test": test_dataset
    }


if __name__ == "__main__":
    """Test the simple Barkopedia gender classification dataset implementation."""
    
    print("=== Testing Simple Barkopedia Gender Classification Dataset ===\n")
    
    # Load HuggingFace token if available
    hf_token = None
    try:
        hf_token = os.environ.get("HF_TOKEN")
    except:
        pass
    
    # Create dataset with cleaning enabled
    print("1. Creating dataset splits with cleaning enabled...")
    splits = create_simple_barkopedia_gender_dataset(
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
