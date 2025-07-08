#!/usr/bin/env python3
"""
Individual Dog Recognition Dataset Implementation (Task 4)
Dataset: ArlingtonCL2/Barkopedia_Individual_Dog_Recognition_Dataset
"""

import os
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from datasets import load_dataset
from transformers import Wav2Vec2FeatureExtractor
from scipy import signal

try:
    from .simple_dataset_interface import SimpleBarkopediaDataset, SimpleDatasetConfig
except ImportError:
    # If running as main, use absolute import
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "barkopedia_datasets"))
    from simple_dataset_interface import SimpleBarkopediaDataset, SimpleDatasetConfig

class BarkopediaIndividualDogDataset(SimpleBarkopediaDataset):
    """
    Individual Dog Recognition Dataset for Barkopedia.
    
    Task: Recognize individual dogs (60 dogs, IDs 1-60)
    - Training: 7,137 clips (~120 clips per 60 dogs)
    - Test: 1,787 clips (~30 clips per 60 dogs)
    """
    
    def __init__(self, config: SimpleDatasetConfig, split: str = "train"):
        super().__init__(config, split)
        
        # Dog ID mapping (1-60 -> 0-59 for model)
        self.num_dogs = 60
        self.id_to_label = {i: f"dog_{i+1}" for i in range(self.num_dogs)}
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        
        # Set up feature extractor for Wav2Vec2
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        
        # HuggingFace dataset name
        self.hf_dataset_name = "ArlingtonCL2/Barkopedia_Individual_Dog_Recognition_Dataset"
    
    def load_data(self) -> None:
        """Load the Individual Dog Recognition dataset."""
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
            
            # Load correct labels from CSV files and map to the correct HF dataset split
            correct_labels_map = None
            raw_data = None
            
            if self.split == "train":
                # Train split: Use train split from HF dataset + train_labels.csv
                try:
                    from huggingface_hub import hf_hub_download
                    import pandas as pd
                    
                    # Download the correct labels CSV
                    csv_file = hf_hub_download(
                        repo_id=self.hf_dataset_name,
                        filename='train_labels.csv',
                        repo_type='dataset',
                        token=self.config.hf_token,
                        cache_dir=self.config.cache_dir
                    )
                    
                    # Load the CSV with correct labels
                    df = pd.read_csv(csv_file)
                    print(f"Loaded correct train labels from CSV: {df.shape[0]} samples")
                    print(f"Train CSV columns: {list(df.columns)}")
                    print(f"Train CSV pred_dog_id distribution: {df['pred_dog_id'].value_counts().head(10).to_dict()}")
                    
                    # Create mapping from audio_id to correct label
                    correct_labels_map = {}
                    missing_count = 0
                    for _, row in df.iterrows():
                        audio_id = row['audio_id']
                        dog_id = row['pred_dog_id']
                        # Map dog_id to 0-based indexing (1-60 -> 0-59)
                        if pd.isna(dog_id) or dog_id is None:
                            missing_count += 1
                            continue
                        label_id = int(dog_id) - 1  # Convert to 0-based
                        if label_id < 0 or label_id >= self.num_dogs:
                            print(f"Warning: Invalid pred_dog_id {dog_id}, skipping...")
                            missing_count += 1
                            continue
                        correct_labels_map[audio_id] = label_id
                    
                    print(f"Created train label mapping for {len(correct_labels_map)} audio files")
                    if missing_count > 0:
                        print(f"Warning: {missing_count} samples have missing/invalid labels in train CSV")
                    
                    # Use train split from HF dataset
                    if "train" in available_splits:
                        raw_data = ds["train"]
                        print(f"Using 'train' split from HF dataset ({len(raw_data)} samples)")
                    else:
                        raise ValueError(f"Train split not found in HF dataset. Available: {available_splits}")
                        
                except Exception as e:
                    print(f"ERROR: Could not load train labels from CSV: {e}")
                    raise
            
            elif self.split == "validation":
                # Validation split: Use validation split from HF dataset + validation_labels.csv
                try:
                    from huggingface_hub import hf_hub_download
                    import pandas as pd
                    
                    # Download the correct validation labels CSV
                    csv_file = hf_hub_download(
                        repo_id=self.hf_dataset_name,
                        filename='validation_labels.csv',
                        repo_type='dataset',
                        token=self.config.hf_token,
                        cache_dir=self.config.cache_dir
                    )
                    
                    # Load the CSV with correct labels
                    df = pd.read_csv(csv_file)
                    print(f"Loaded correct validation labels from CSV: {df.shape[0]} samples")
                    print(f"Validation CSV columns: {list(df.columns)}")
                    print(f"Validation CSV pred_dog_id distribution: {df['pred_dog_id'].value_counts().head(10).to_dict()}")
                    
                    # Create mapping from audio_id to correct label
                    correct_labels_map = {}
                    missing_count = 0
                    for _, row in df.iterrows():
                        audio_id = row['audio_id']
                        dog_id = row['pred_dog_id']
                        # Map dog_id to 0-based indexing (1-60 -> 0-59)
                        if pd.isna(dog_id) or dog_id is None:
                            missing_count += 1
                            continue
                        label_id = int(dog_id) - 1  # Convert to 0-based
                        if label_id < 0 or label_id >= self.num_dogs:
                            print(f"Warning: Invalid pred_dog_id {dog_id}, skipping...")
                            missing_count += 1
                            continue
                        correct_labels_map[audio_id] = label_id
                    
                    print(f"Created validation label mapping for {len(correct_labels_map)} audio files")
                    if missing_count > 0:
                        print(f"Warning: {missing_count} samples have missing/invalid labels in validation CSV")
                    
                    # Use validation split from HF dataset
                    if "validation" in available_splits:
                        raw_data = ds["validation"]
                        print(f"Using 'validation' split from HF dataset ({len(raw_data)} samples)")
                    else:
                        raise ValueError(f"Validation split not found in HF dataset. Available: {available_splits}")
                        
                except Exception as e:
                    print(f"ERROR: Could not load validation labels from CSV: {e}")
                    raise
            
            elif self.split == "test":
                # Test split: Use test split from HF dataset (no labels, for submission)
                print("Loading test split for inference/submission (no labels)")
                if "test" in available_splits:
                    raw_data = ds["test"]
                    print(f"Using 'test' split from HF dataset ({len(raw_data)} samples)")
                    correct_labels_map = None  # No labels for test split
                else:
                    raise ValueError(f"Test split not found in HF dataset. Available: {available_splits}")
            
            else:
                raise ValueError(f"Invalid split '{self.split}'. Must be one of: train, validation, test")
            
            print(f"Processing {len(raw_data)} samples...")
            
            # Cast audio column to disable automatic decoding
            from datasets import Audio
            raw_data = raw_data.cast_column("audio", Audio(decode=False))
            
            # Process each sample with robust error handling
            self.data = []
            self.labels = []
            unique_labels = set()
            processed_count = 0
            skipped_count = 0
            total_segments = 0
            missing_labels = []
            
            for i in range(len(raw_data)):
                try:
                    # Get sample - audio will now be a path/bytes, not pre-decoded
                    sample_dict = raw_data[i].copy()  # Make a copy to avoid modifying original
                    
                    # For train and validation splits, we need to inject the correct label from CSV
                    if self.split in ["train", "validation"] and correct_labels_map is not None:
                        # Extract audio filename to get the correct label
                        audio_info = sample_dict["audio"]
                        if isinstance(audio_info, dict) and "path" in audio_info:
                            audio_path = audio_info["path"]
                        elif isinstance(audio_info, str):
                            audio_path = audio_info
                        else:
                            # Skip if we can't get the path
                            skipped_count += 1
                            if skipped_count <= 10:
                                print(f"Skipping sample {i+1}: Cannot extract audio path for label lookup...")
                            elif skipped_count == 11:
                                print("... (suppressing further error messages)")
                            continue
                        
                        # Extract filename without extension
                        import os
                        audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
                        
                        # Look up the correct label in CSV mapping
                        if audio_filename in correct_labels_map:
                            # Inject the correct label into the sample
                            sample_dict["label"] = correct_labels_map[audio_filename]
                        else:
                            # Track missing labels for debugging
                            missing_labels.append(audio_filename)
                            skipped_count += 1
                            if skipped_count <= 10:
                                print(f"Skipping sample {i+1}: No label found for {audio_filename}")
                            elif skipped_count == 11:
                                print("... (suppressing further error messages)")
                            continue
                    
                    elif self.split == "test":
                        # For test split, set a dummy label since we don't have real labels
                        sample_dict["label"] = 0  # Dummy label, will be ignored during inference
                    
                    # Now process the sample (which should have labels)
                    processed_segments = self._process_sample(sample_dict)
                    
                    # Add all segments from this sample
                    for segment in processed_segments:
                        self.data.append(segment)
                        self.labels.append(segment['labels'])
                        if self.split != "test":  # Don't track dummy labels for test split
                            unique_labels.add(segment['labels'])
                        total_segments += 1
                    
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} audio files -> {total_segments} segments (skipped: {skipped_count})...")
                        
                except Exception as e:
                    # Skip corrupted samples
                    skipped_count += 1
                    if skipped_count <= 10:  # Only print first 10 errors to avoid spam
                        print(f"Skipping corrupted sample {i+1}: {str(e)[:100]}...")
                    elif skipped_count == 11:
                        print("... (suppressing further error messages)")
                    continue
            
            # Print debugging info about missing labels
            if missing_labels:
                print(f"WARNING: {len(missing_labels)} audio files had no labels in CSV mapping")
                if len(missing_labels) <= 20:
                    print(f"Missing CSV labels for: {missing_labels}")
                else:
                    print(f"First 20 missing CSV labels: {missing_labels[:20]}")
                    print(f"... and {len(missing_labels) - 20} more")
            
            # Update label mapping based on discovered labels
            if self.split != "test":  # Don't use dummy labels for mapping
                unique_labels = sorted(list(unique_labels))
                print(f"Found unique dog IDs: {len(unique_labels)} (expected: {self.num_dogs})")
                if unique_labels:
                    print(f"Dog ID range: {min(unique_labels)} - {max(unique_labels)}")
            
            # For individual dog recognition, always use standard mapping (0-59 for dogs 1-60)
            self.id_to_label = {i: f"dog_{i+1}" for i in range(self.num_dogs)}
            self.label_to_id = {v: k for k, v in self.id_to_label.items()}
            print(f"Label mapping: 60 dogs (IDs 1-60 mapped to indices 0-59)")
            
            # Print segmentation summary
            if self.config.enable_segmentation:
                print(f"Segmentation enabled: {processed_count} audio files -> {total_segments} segments")
                print(f"Average segments per file: {total_segments/max(processed_count, 1):.2f}")
            
            print(f"Loaded {len(self.data)} samples for {self.split} split (skipped: {skipped_count} files)")
            if self.split != "test":
                print(f"Dog distribution: {self.get_class_distribution()}")
            
            if len(self.data) == 0:
                raise RuntimeError("No valid samples found in dataset after filtering corrupted files")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with Wav2Vec2 feature extraction."""
        # Get the raw data
        sample = self.data[idx]
        
        # Extract Wav2Vec2 features
        audio = sample["audio"]
        input_values = self.feature_extractor(
            audio,
            sampling_rate=self.config.sampling_rate,
            return_tensors="np"
        )["input_values"].squeeze()
        
        # Return sample with both raw audio and processed features
        return {
            "audio": audio,
            "input_values": input_values,
            "labels": sample["labels"],
            "label_name": sample["label_name"],
            "sampling_rate": sample["sampling_rate"],
            "segment_id": sample.get("segment_id", 0),
            "source_file": sample.get("source_file", "unknown"),
            "segment_start": sample.get("segment_start", 0.0),
            "segment_duration": sample.get("segment_duration", len(audio) / self.config.sampling_rate),
            "dog_id": sample["labels"] + 1  # Convert back to 1-60 for human readability
        }
    
    def _process_sample(self, sample: Any) -> List[Dict[str, Any]]:
        """Process a single sample into the format expected by models."""
        # Extract audio data
        audio_info = sample["audio"]
        
        if isinstance(audio_info, dict):
            if "array" in audio_info and audio_info["array"] is not None:
                audio = audio_info["array"]
                sampling_rate = audio_info["sampling_rate"]
                audio_path = audio_info.get("path", "unknown")
            elif "path" in audio_info:
                audio_path = audio_info["path"]
                try:
                    import soundfile as sf
                    audio, sampling_rate = sf.read(audio_path)
                except Exception as e:
                    raise ValueError(f"Could not load audio from {audio_path}: {e}")
            elif "bytes" in audio_info:
                try:
                    import soundfile as sf
                    import io
                    audio, sampling_rate = sf.read(io.BytesIO(audio_info["bytes"]))
                    audio_path = "bytes_data"
                except Exception as e:
                    raise ValueError(f"Could not load audio from bytes: {e}")
            else:
                raise ValueError(f"Unknown audio format: {list(audio_info.keys())}")
        else:
            try:
                import soundfile as sf
                if isinstance(audio_info, str):
                    audio_path = audio_info
                    audio, sampling_rate = sf.read(audio_info)
                else:
                    import io
                    audio, sampling_rate = sf.read(io.BytesIO(audio_info))
                    audio_path = "bytes_data"
            except Exception as e:
                raise ValueError(f"Could not load audio: {e}")
        
        # Convert to numpy array and ensure correct format
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        audio = audio.astype(np.float32)
        
        # Convert stereo to mono if needed
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        # Ensure audio is 1D
        if len(audio.shape) > 1:
            audio = audio.flatten()
        
        # Minimum length check for Wav2Vec2
        min_length = 1600  # 0.1 seconds at 16kHz
        if len(audio) < min_length:
            padding_needed = min_length - len(audio)
            audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)
        
        # Resample if needed
        if sampling_rate != self.config.sampling_rate:
            audio = self.resample_audio(audio, sampling_rate, self.config.sampling_rate)
        
        # Apply cleaning if enabled
        audio = self.clean_audio(audio, self.config.sampling_rate)
        
        # Get label information
        if "label" in sample:
            label_id = sample["label"]
        elif "labels" in sample:
            label_id = sample["labels"]
        else:
            raise ValueError(f"No label found in sample. Available keys: {list(sample.keys())}")
        
        # Validate label range (0-59 for 60 dogs)
        if label_id < 0 or label_id >= self.num_dogs:
            raise ValueError(f"Invalid dog ID {label_id}, must be 0-{self.num_dogs-1}")
        
        label_name = self.id_to_label.get(label_id, f"dog_{label_id+1}")
        
        # Apply segmentation if enabled
        if self.config.enable_segmentation:
            return self._segment_audio(audio, self.config.sampling_rate, label_id, audio_path)
        else:
            return [{
                "audio": audio,
                "labels": label_id,
                "label_name": label_name,
                "sampling_rate": self.config.sampling_rate,
                "segment_id": 0,
                "source_file": audio_path,
                "segment_start": 0.0,
                "segment_duration": len(audio) / self.config.sampling_rate
            }]


def create_individual_dog_dataset(
    cache_dir: str = "./barkopedia_dataset",
    hf_token: Optional[str] = None,
    apply_cleaning: bool = False,
    sampling_rate: int = 16000,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
    enable_segmentation: bool = False,
    segment_duration: float = 2.0,
    segment_overlap: float = 0.1,
    energy_threshold: float = 0.01
) -> Dict[str, BarkopediaIndividualDogDataset]:
    """
    Create Individual Dog Recognition dataset splits.
    
    Returns:
        Dictionary with 'train', 'validation', and 'test' dataset instances
    """
    
    # Create configuration
    config = SimpleDatasetConfig(
        dataset_name="barkopedia_individual_dog",
        cache_dir=cache_dir,
        hf_token=hf_token,
        sampling_rate=sampling_rate,
        apply_cleaning=apply_cleaning,
        max_duration=max_duration,
        min_duration=min_duration,
        enable_segmentation=enable_segmentation,
        segment_duration=segment_duration,
        segment_overlap=segment_overlap,
        energy_threshold=energy_threshold
    )
    
    # Create train and validation datasets (validation split from train)
    train_dataset = BarkopediaIndividualDogDataset(config, split="train")
    train_dataset.load_data()
    
    validation_dataset = BarkopediaIndividualDogDataset(config, split="validation")
    validation_dataset.load_data()
    
    # Create test dataset
    test_dataset = BarkopediaIndividualDogDataset(config, split="test")
    test_dataset.load_data()
    
    return {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    }


if __name__ == "__main__":
    """Test the Individual Dog Recognition dataset implementation."""
    
    print("=== Testing Individual Dog Recognition Dataset ===\n")
    
    # Load HuggingFace token if available
    hf_token = None
    try:
        hf_token = os.environ.get("HF_TOKEN")
    except:
        pass
    
    # Create dataset
    print("1. Creating dataset splits...")
    splits = create_individual_dog_dataset(
        hf_token=hf_token,
        apply_cleaning=True,
        enable_segmentation=False  # Start without segmentation for testing
    )
    
    train_dataset = splits["train"]
    val_dataset = splits["validation"]
    test_dataset = splits["test"]
    
    print(f"   ✓ Train dataset: {len(train_dataset)} samples")
    print(f"   ✓ Validation dataset: {len(val_dataset)} samples")
    print(f"   ✓ Test dataset: {len(test_dataset)} samples")
    
    # Test sample access
    print("\n2. Testing sample access...")
    if len(train_dataset) > 0:
        train_sample = train_dataset[0]
        print(f"   ✓ Sample keys: {list(train_sample.keys())}")
        print(f"   ✓ Dog ID: {train_sample['dog_id']} (label: {train_sample['labels']})")
        print(f"   ✓ Label name: {train_sample['label_name']}")
        print(f"   ✓ Input values shape: {train_sample['input_values'].shape}")
        print(f"   ✓ Audio shape: {train_sample['audio'].shape}")
        print(f"   ✓ Sampling rate: {train_sample['sampling_rate']}")
    
    # Test class distribution
    print("\n3. Class distribution:")
    train_dist = train_dataset.get_class_distribution()
    print(f"   Train distribution (top 10): {dict(list(train_dist.items())[:10])}")
    print(f"   Total unique dogs in train: {len(train_dist)}")
    
    if len(val_dataset) > 0:
        val_dist = val_dataset.get_class_distribution()
        print(f"   Validation distribution (top 10): {dict(list(val_dist.items())[:10])}")
        print(f"   Total unique dogs in validation: {len(val_dist)}")
    
    # Test DataLoader
    print("\n4. Testing DataLoader...")
    if len(train_dataset) > 0:
        dataloader = train_dataset.get_dataloader(batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        
        print(f"   ✓ Batch input_values shape: {batch['input_values'].shape}")
        print(f"   ✓ Batch labels shape: {batch['labels'].shape}")
        print(f"   ✓ Labels in batch: {batch['labels']}")
        print(f"   ✓ Dog IDs in batch: {[train_dataset.id_to_label[label.item()] for label in batch['labels']]}")
    
    print("\n=== All tests passed! ===")
