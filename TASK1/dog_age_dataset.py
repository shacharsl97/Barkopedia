#!/usr/bin/env python3
"""
Dog Age Group Classification Dataset Implementation
Dataset: ArlingtonCL2/Barkopedia_DOG_AGE_GROUP_CLASSIFICATION_DATASET
"""
from typing import Dict, Any, Optional, List
import os
import numpy as np
import pandas as pd
from datasets import load_dataset, Audio
from transformers import Wav2Vec2FeatureExtractor

try:
    from .simple_dataset_interface import SimpleBarkopediaDataset, SimpleDatasetConfig
except ImportError:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "barkopedia_datasets"))
    from simple_dataset_interface import SimpleBarkopediaDataset, SimpleDatasetConfig

class BarkopediaDogAgeDataset(SimpleBarkopediaDataset):
    """
    Dog Age Group Classification Dataset for Barkopedia.
    5 classes: 0-4 (see dataset for mapping)
    """
    def __init__(self, config: SimpleDatasetConfig, split: str = "train", backbone: str = "ast"):
        self.hf_dataset_name = "ArlingtonCL2/Barkopedia_DOG_AGE_GROUP_CLASSIFICATION_DATASET"
        self.n_classes = 5
        self.id_to_label = {0: "puppy", 1: "juvenile", 2: "adolescents", 3: "adult", 4: "senior"}
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.backbone = backbone
        if backbone == "ast":
            from transformers import ASTFeatureExtractor
            self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        elif backbone == "huber":
            from transformers import Wav2Vec2FeatureExtractor
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        elif backbone == "wav2vec2":
            from transformers import Wav2Vec2FeatureExtractor
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h")
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        super().__init__(config, split)

    def preprocess(self, audio_array, sr):
        import numpy as np
        MAX_AUDIO_LEN = 50000  # ~3.1 seconds at 16kHz
        if self.backbone == 'wav2vec2' and len(audio_array) > MAX_AUDIO_LEN:
            audio_array = audio_array[:MAX_AUDIO_LEN]
        # Optionally add more preprocessing per backbone here
        return audio_array, sr

    def load_data(self) -> None:
        print(f"Loading {self.split} data from HuggingFace Hub...")
        self.n_classes = 5
        self.id_to_label = {0: "puppy", 1: "juvenile", 2: "adolescents", 3: "adult", 4: "senior"}
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        try:
            ds = load_dataset(
                self.hf_dataset_name,
                token=self.config.hf_token,
                cache_dir=self.config.cache_dir
            )
            available_splits = list(ds.keys())
            print(f"Available splits: {available_splits}")
            # Load correct labels from CSV if available
            correct_labels_map = None
            import pandas as pd
            from huggingface_hub import hf_hub_download
            if self.split != "test":
                try:
                    csv_file = hf_hub_download(
                        repo_id=self.hf_dataset_name,
                        filename=f'{self.split}_labels.csv',
                        repo_type='dataset',
                        token=self.config.hf_token,
                        cache_dir=self.config.cache_dir
                    )
                    df = pd.read_csv(csv_file)
                    print(f"Loaded correct {self.split} labels from CSV: {df.shape[0]} samples")
                    # Create mapping from audio_id to correct label
                    correct_labels_map = {}
                    missing_count = 0
                    for _, row in df.iterrows():
                        print()
                        audio_id = row['audio_id']
                        age = row['pred_dog_age_group'] if self.split == "train" else row['pred_dog_id']
                        # Map dog_id to 0-based indexing (1-60 -> 0-59)
                        if pd.isna(age) or age is None:
                            missing_count += 1
                            continue
                        label_id = self.label_to_id[age]
                        if label_id < 0 or label_id >= 5:
                            print(f"Warning: Invalid pred_age {age} {label_id}, skipping...")
                            missing_count += 1
                            continue
                        correct_labels_map[audio_id] = label_id
                except Exception as e:
                    print(f"ERROR: Could not load {self.split} labels from CSV: {e}")
                    correct_labels_map = None
            elif self.split == "test":
                correct_labels_map = None
            # Select split
            if self.split == "train" and len(available_splits) > 0:
                raw_data = ds[available_splits[0]]
            elif self.split == "validation" and len(available_splits) > 1:
                raw_data = ds[available_splits[1]]
            elif self.split == "test" and len(available_splits) > 2:
                raw_data = ds[available_splits[2]]
            else:
                raw_data = ds[available_splits[0]]
                print(f"Warning: {self.split} split not found, using {available_splits[0]}")
            print(f"Processing {len(raw_data)} samples...")
            raw_data = raw_data.cast_column("audio", Audio(decode=True))
            self.data = []
            self.labels = []
            skipped = 0
            for idx, sample in enumerate(raw_data):
                try:
                    if correct_labels_map is not None and self.split != "test":
                        audio_info = sample["audio"]
                        if isinstance(audio_info, dict) and "path" in audio_info:
                            audio_path = audio_info["path"]
                        elif isinstance(audio_info, str):
                            audio_path = audio_info
                        else:
                            audio_path = None
                        if audio_path is not None:
                            import os
                            audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
                            if audio_filename in correct_labels_map:
                                sample["label"] = correct_labels_map[audio_filename]
                            else:
                                skipped += 1
                                if skipped <= 10:
                                    print(f"[Warning] Skipping sample {idx} (no label in CSV for {audio_filename})")
                                elif skipped == 11:
                                    print("[Warning] ... further warnings suppressed ...")
                                continue
                    elif self.split == "test":
                        sample["label"] = 0 # Dummy label for test
                    else:
                        if correct_labels_map is None:
                            print("Correct labels map is None")
                        else:
                            print(f"Unexpected split {self.split}")
                        exit(1)
                    processed_sample = self._process_sample(sample)
                    self.data.append(processed_sample)
                    self.labels.append(processed_sample['labels'])
                except Exception as e:
                    skipped += 1
                    if skipped <= 10:
                        print(f"[Warning] Skipping sample {idx} due to error: {e}")
                    elif skipped == 11:
                        print("[Warning] ... further warnings suppressed ...")
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(raw_data)} samples (skipped: {skipped})")
            print(f"Loaded {len(self.data)} samples for {self.split} split (skipped: {skipped})")
            print(f"Label distribution: {self.get_class_distribution()}")
            if len(self.data) == 0:
                raise RuntimeError("No valid samples found in dataset after filtering corrupted files")
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
            sampling_rate = self.config.sampling_rate
        audio = self.clean_audio(audio, sampling_rate)
        audio, sampling_rate = self.preprocess(audio, sampling_rate)
        label_id = sample["label"]
        label_name = self.id_to_label.get(label_id, str(label_id))
        import librosa
        audio_array = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        audio_array, _ = librosa.effects.trim(audio_array)
        audio_array = librosa.util.normalize(audio_array)
        sampling_rate = 16000

        inputs = self.feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )
        return {
                "input_values": inputs.input_values[0],
                "audio": audio,
                "labels": label_id,
                "label_name": label_name,
                "sampling_rate": sampling_rate
            }

import pickle
import hashlib

def get_cache_path(dataset_name, split, backbone, sampling_rate, apply_cleaning, max_duration, min_duration, cache_dir):
    config_str = f"{dataset_name}_{split}_{backbone}_{sampling_rate}_{apply_cleaning}_{max_duration}_{min_duration}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    cache_folder = os.path.join(cache_dir, dataset_name)
    os.makedirs(cache_folder, exist_ok=True)
    return os.path.join(cache_folder, f"{split}_{config_hash}.pkl")

def create_dog_age_dataset(
    cache_dir: str = "./barkopedia_dataset",
    hf_token: Optional[str] = None,
    apply_cleaning: bool = False,
    sampling_rate: int = 16000,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
    backbone: str = "ast"
) -> Dict[str, BarkopediaDogAgeDataset]:
    config = SimpleDatasetConfig(
        dataset_name="barkopedia_dog_age",
        cache_dir=cache_dir,
        hf_token=hf_token,
        sampling_rate=sampling_rate,
        apply_cleaning=apply_cleaning,
        max_duration=max_duration,
        min_duration=min_duration
    )
    datasets = {}
    for split in ["train", "validation", "test"]:
        cache_path = get_cache_path(
            config.dataset_name, split, backbone, sampling_rate, apply_cleaning, max_duration, min_duration, cache_dir
        )
        if os.path.exists(cache_path):
            print(f"[Cache] Loading {split} from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                datasets[split] = pickle.load(f)
        else:
            print(f"[Cache] No cache found for {split}. Preprocessing and caching to {cache_path} ...")
            ds = BarkopediaDogAgeDataset(config, split=split, backbone=backbone)
            ds.load_data()
            with open(cache_path, 'wb') as f:
                pickle.dump(ds, f)
            print(f"[Cache] Finished preprocessing and cached {split} at {cache_path}")
            datasets[split] = ds
    return datasets

if __name__ == "__main__":
    print("=== Testing Dog Age Group Classification Dataset ===\n")
    hf_token = None
    try:
        hf_token = os.environ.get("HF_TOKEN")
    except:
        pass
    print("1. Creating dataset splits...")
    splits = create_dog_age_dataset(
        hf_token=hf_token,
        apply_cleaning=True
    )
    train_dataset = splits["train"]
    test_dataset = splits["test"]
    print(f"   ✓ Train dataset: {len(train_dataset)} samples")
    print(f"   ✓ Test dataset: {len(test_dataset)} samples")
    print("\n2. Testing sample access...")
    if len(train_dataset) > 0:
        train_sample = train_dataset[0]
        print(f"   ✓ Sample keys: {list(train_sample.keys())}")
        print(f"   ✓ Label: {train_sample['labels']} ({train_sample['label_name']})")
        print(f"   ✓ Input values shape: {train_sample.get('input_values', np.array([])).shape}")
        print(f"   ✓ Audio shape: {train_sample['audio'].shape}")
        print(f"   ✓ Sampling rate: {train_sample['sampling_rate']}")
    print("\n3. Class distribution:")
    train_dist = train_dataset.get_class_distribution()
    print(f"   Train distribution: {dict(train_dist)}")
    print(f"   Total unique classes in train: {len(train_dist)}")
    print("\n4. Testing DataLoader...")
    if len(train_dataset) > 0:
        dataloader = train_dataset.get_dataloader(batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        print(f"   ✓ Batch input_values shape: {batch['input_values'].shape if 'input_values' in batch else 'N/A'}")
        print(f"   ✓ Batch labels shape: {batch['labels'].shape}")
        print(f"   ✓ Labels in batch: {batch['labels']}")
    print("\n=== All tests passed! ===")
