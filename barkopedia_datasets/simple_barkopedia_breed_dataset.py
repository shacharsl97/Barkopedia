import os
import numpy as np
import torch
from typing import Dict, Any, Optional
from datasets import load_dataset
from transformers import ASTFeatureExtractor

try:
    from .simple_dataset_interface import SimpleBarkopediaDataset, SimpleDatasetConfig
except ImportError:
    from simple_dataset_interface import SimpleBarkopediaDataset, SimpleDatasetConfig

class SimpleBarkopediaBreedDataset(SimpleBarkopediaDataset):
    """
    Simple implementation for Barkopedia Dog Breed Classification Dataset.
    
    Breed groups: 0-4 (replace with actual breed names if available)
    """
    def __init__(self, config: SimpleDatasetConfig, split: str = "train"):
        super().__init__(config, split)
        # Example mapping, replace with actual breed names if available
        self.id_to_label = {
            0: "chiuaua",
            1: "german shepherd",
            2: "husky",
            3: "pitbull",
            4: "shiba inu"
        }
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.hf_dataset_name = "ArlingtonCL2/Barkopedia_DOG_BREED_CLASSIFICATION_DATASET"
    def load_data(self) -> None:
        print(f"Loading {self.split} data from HuggingFace Hub...")
        hf_token_path = os.path.join(os.path.dirname(__file__), 'hf_token.py')
        if os.path.exists(hf_token_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location('hf_token', hf_token_path)
            hf_token_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hf_token_module)
            HF_TOKEN = getattr(hf_token_module, 'HF_TOKEN', None)
        else:
            HF_TOKEN = os.environ.get("HF_TOKEN")
        try:
            ds = load_dataset(
                self.hf_dataset_name,
                token=HF_TOKEN,
                # cache_dir=self.config.cache_dir
            )
            available_splits = list(ds.keys())
            print(f"Available splits: {available_splits}")
            if self.split == "train" and len(available_splits) > 0:
                raw_data = ds[available_splits[0]]
            elif self.split == "test" and len(available_splits) > 1:
                raw_data = ds[available_splits[1]]
            else:
                raw_data = ds[available_splits[0]]
                print(f"Warning: {self.split} split not found, using {available_splits[0]}")
            print(f"Processing {len(raw_data)} samples...")
            self.data = []
            self.labels = []
            skipped = 0
            for i, sample in enumerate(raw_data):
                try:
                    processed_sample = self._process_sample(sample)
                    self.data.append(processed_sample)
                    self.labels.append(processed_sample['labels'])
                except Exception as e:
                    skipped += 1
                    print(f"[Warning] Skipping sample {i} due to error: {e}")
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(raw_data)} samples (skipped: {skipped})")
            print(f"Loaded {len(self.data)} samples for {self.split} split (skipped: {skipped})")
            print(f"Label distribution: {self.get_class_distribution()}")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    def _process_sample(self, sample: Any) -> Dict[str, Any]:
        if isinstance(sample["audio"], dict):
            audio = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
        else:
            raise ValueError("Unexpected audio format in sample")
        if sampling_rate != self.config.sampling_rate:
            audio = self.resample_audio(audio, sampling_rate, self.config.sampling_rate)
        audio = self.clean_audio(audio, self.config.sampling_rate)
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=self.config.sampling_rate, 
            return_tensors="pt"
        )
        input_values = inputs.input_values.squeeze(0)
        label_id = sample["label"]
        label_name = self.id_to_label.get(label_id, str(label_id))
        return {
            "input_values": input_values,
            "labels": label_id,
            "label_name": label_name,
            "sampling_rate": self.config.sampling_rate,
            "audio": audio
        }

def create_simple_barkopedia_breed_dataset(
    cache_dir: str = "./barkopedia_dataset",
    hf_token: Optional[str] = None,
    apply_cleaning: bool = False,
    sampling_rate: int = 16000,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None
) -> Dict[str, SimpleBarkopediaBreedDataset]:
    """
    Create simple Barkopedia dog breed classification dataset splits.
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
    config = SimpleDatasetConfig(
        dataset_name="barkopedia_breed",
        cache_dir=cache_dir,
        hf_token=hf_token,
        sampling_rate=sampling_rate,
        apply_cleaning=apply_cleaning,
        max_duration=max_duration,
        min_duration=min_duration
    )
    train_dataset = SimpleBarkopediaBreedDataset(config, split="train")
    train_dataset.load_data()
    test_dataset = SimpleBarkopediaBreedDataset(config, split="test")
    test_dataset.load_data()
    return {
        "train": train_dataset,
        "test": test_dataset
    }

if __name__ == "__main__":
    print("=== Testing Simple Barkopedia Breed Classification Dataset ===\n")
    hf_token = None
    try:
        hf_token = os.environ.get("HF_TOKEN")
    except:
        pass
    print("1. Creating dataset splits with cleaning enabled...")
    splits = create_simple_barkopedia_breed_dataset(
        hf_token=hf_token,
        apply_cleaning=True,
        max_duration=5.0
    )
    train_dataset = splits["train"]
    test_dataset = splits["test"]
    print(f"   ✓ Train dataset: {len(train_dataset)} samples")
    print(f"   ✓ Test dataset: {len(test_dataset)} samples")
    print("\n2. Testing sample access...")
    train_sample = train_dataset[0]
    print(f"   ✓ Sample keys: {list(train_sample.keys())}")
    print(f"   ✓ Input values shape: {train_sample['input_values'].shape}")
    print(f"   ✓ Label: {train_sample['labels']} ({train_sample['label_name']})")
    print(f"   ✓ Sampling rate: {train_sample['sampling_rate']}")
    print(f"   ✓ Audio shape: {train_sample['audio'].shape}")
    print("\n3. Class distribution:")
    train_dist = train_dataset.get_class_distribution()
    for label, count in train_dist.items():
        print(f"   {label}: {count} samples")
    print("\n4. Testing DataLoader...")
    dataloader = train_dataset.get_dataloader(batch_size=4, shuffle=False)
    batch = next(iter(dataloader))
    print(f"   ✓ Batch input_values shape: {batch['input_values'].shape}")
    print(f"   ✓ Batch labels shape: {batch['labels'].shape}")
    print(f"   ✓ Labels in batch: {batch['labels']}")
    print("\n=== All tests passed! ===")
