import os
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from datasets import load_dataset
from transformers import ASTFeatureExtractor
from scipy import signal

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
            
            # Extract labels from folder structure instead of CSV files
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
                    print(f"Train CSV label distribution: {df['pred_dog_sex'].value_counts().to_dict()}")
                    
                    # Create mapping from audio_id to correct label
                    correct_labels_map = {}
                    missing_count = 0
                    for _, row in df.iterrows():
                        audio_id = row['audio_id']
                        gender = row['pred_dog_sex']
                        # Map gender string to numeric label
                        if pd.isna(gender) or gender is None:
                            missing_count += 1
                            continue
                        label_id = 0 if str(gender).lower() == 'female' else 1
                        correct_labels_map[audio_id] = label_id
                    
                    print(f"Created train label mapping for {len(correct_labels_map)} audio files")
                    if missing_count > 0:
                        print(f"Warning: {missing_count} samples have missing labels in train CSV")
                    
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
                    
                    # Use pred_dog_id column for gender information (note: different column name than train!)
                    gender_col = 'pred_dog_id'
                    if gender_col not in df.columns:
                        raise ValueError(f"Required column '{gender_col}' not found in validation CSV. Available columns: {list(df.columns)}")
                    
                    print(f"Using column '{gender_col}' for validation labels")
                    print(f"Validation CSV label distribution: {df[gender_col].value_counts().to_dict()}")
                    
                    # Create mapping from audio_id to correct label
                    correct_labels_map = {}
                    missing_count = 0
                    for _, row in df.iterrows():
                        audio_id = row['audio_id']
                        gender = row[gender_col]
                        # Map gender string to numeric label
                        if pd.isna(gender) or gender is None:
                            missing_count += 1
                            continue
                        if isinstance(gender, str):
                            label_id = 0 if gender.lower() == 'female' else 1
                        else:
                            # Assume numeric: 0 = female, 1 = male
                            label_id = int(gender)
                        correct_labels_map[audio_id] = label_id
                    
                    print(f"Created validation label mapping for {len(correct_labels_map)} audio files")
                    if missing_count > 0:
                        print(f"Warning: {missing_count} samples have missing labels in validation CSV")
                    
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
            
            # Cast audio column to disable automatic decoding - this prevents corruption errors
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
                    processed_segments = self._process_sample(sample_dict, None)  # No need to pass correct_labels_map anymore
                    
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
                print(f"Found unique labels: {unique_labels}")
            
            # For gender classification, always use standard mapping
            self.id_to_label = {
                0: "female", 
                1: "male"
            }
            
            self.label_to_id = {v: k for k, v in self.id_to_label.items()}
            print(f"Updated label mapping: {self.id_to_label}")
            
            # Print segmentation summary
            if self.config.enable_segmentation:
                print(f"Segmentation enabled: {processed_count} audio files -> {total_segments} segments")
                print(f"Average segments per file: {total_segments/max(processed_count, 1):.2f}")
            
            print(f"Loaded {len(self.data)} samples for {self.split} split (skipped: {skipped_count} files)")
            if self.split != "test":
                print(f"Label distribution: {self.get_class_distribution()}")
            
            if len(self.data) == 0:
                raise RuntimeError("No valid samples found in dataset after filtering corrupted files")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    
    def _detect_silence_boundaries(self, audio: np.ndarray, sampling_rate: int) -> List[Tuple[int, int]]:
        """
        Detect silence boundaries in audio for intelligent segmentation.
        
        Args:
            audio: Audio array
            sampling_rate: Sampling rate
            
        Returns:
            List of (start_idx, end_idx) tuples for non-silent regions
        """
        # Calculate short-time energy
        frame_length = int(0.025 * sampling_rate)  # 25ms frames
        hop_length = int(0.010 * sampling_rate)    # 10ms hop
        
        # Pad audio to ensure we can process it
        if len(audio) < frame_length:
            return [(0, len(audio))]
        
        # Calculate energy for each frame
        energies = []
        for i in range(0, len(audio) - frame_length + 1, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sum(frame ** 2) / len(frame)
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Normalize energy
        if np.max(energies) > 0:
            energies = energies / np.max(energies)
        
        # Detect silence (below threshold)
        silence_mask = energies < self.config.energy_threshold
        
        # Find transitions between silence and non-silence
        transitions = np.diff(silence_mask.astype(int))
        
        # Get start and end points of non-silent regions
        non_silent_regions = []
        
        # Check if we start with non-silence
        if not silence_mask[0]:
            start = 0
        else:
            start_transitions = np.where(transitions == -1)[0] + 1
            start = start_transitions[0] * hop_length if len(start_transitions) > 0 else None
        
        # Find all silence-to-sound and sound-to-silence transitions
        silence_to_sound = np.where(transitions == -1)[0] + 1
        sound_to_silence = np.where(transitions == 1)[0] + 1
        
        # Build regions
        if start is not None:
            current_start_frame = 0 if not silence_mask[0] else silence_to_sound[0] if len(silence_to_sound) > 0 else None
            
            if current_start_frame is not None:
                current_start = current_start_frame * hop_length
                
                for end_frame in sound_to_silence:
                    end = end_frame * hop_length
                    duration = (end - current_start) / sampling_rate
                    
                    # Only include regions longer than minimum silence duration
                    if duration >= self.config.silence_min_duration:
                        non_silent_regions.append((current_start, min(end, len(audio))))
                    
                    # Find next start
                    next_starts = silence_to_sound[silence_to_sound > end_frame]
                    if len(next_starts) > 0:
                        current_start = next_starts[0] * hop_length
                    else:
                        break
                
                # Handle final region if it doesn't end in silence
                if len(sound_to_silence) == 0 or (len(silence_to_sound) > 0 and 
                    (len(sound_to_silence) == 0 or silence_to_sound[-1] > sound_to_silence[-1])):
                    non_silent_regions.append((current_start, len(audio)))
        
        # If no regions found, return the entire audio
        if not non_silent_regions:
            non_silent_regions = [(0, len(audio))]
        
        return non_silent_regions
    
    def _segment_audio(self, audio: np.ndarray, sampling_rate: int, label_id: int, 
                      audio_path: str = "unknown") -> List[Dict[str, Any]]:
        """
        Segment long audio into multiple chunks of target duration.
        
        Args:
            audio: Audio array
            sampling_rate: Sampling rate
            label_id: Label for all segments
            audio_path: Path to original audio file
            
        Returns:
            List of segmented audio samples
        """
        duration = len(audio) / sampling_rate
        target_samples = int(self.config.segment_duration * sampling_rate)
        overlap_samples = int(self.config.segment_overlap * sampling_rate)
        min_samples = int(self.config.min_segment_duration * sampling_rate)
        max_samples = int(self.config.max_segment_duration * sampling_rate)
        
        # For Wav2Vec2 compatibility, ensure minimum 1 second segments
        wav2vec2_min_duration = 1.0  # 1 second minimum for Wav2Vec2
        wav2vec2_min_samples = int(wav2vec2_min_duration * sampling_rate)
        effective_min_samples = max(min_samples, wav2vec2_min_samples)
        effective_min_duration = effective_min_samples / sampling_rate
        
        # If audio is shorter than effective minimum duration, pad it
        if duration < effective_min_duration:
            padding_needed = effective_min_samples - len(audio)
            audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)
            duration = len(audio) / sampling_rate
            
            return [{
                "audio": audio,
                "labels": label_id,
                "label_name": self.id_to_label.get(label_id, f"unknown_{label_id}"),
                "sampling_rate": sampling_rate,
                "segment_id": 0,
                "source_file": audio_path,
                "segment_start": 0.0,
                "segment_duration": duration
            }]
        
        # If audio is shorter than target duration but longer than minimum, return as-is
        if duration <= self.config.segment_duration:
            return [{
                "audio": audio,
                "labels": label_id,
                "label_name": self.id_to_label.get(label_id, f"unknown_{label_id}"),
                "sampling_rate": sampling_rate,
                "segment_id": 0,
                "source_file": audio_path,
                "segment_start": 0.0,
                "segment_duration": duration
            }]
        
        segments = []
        
        # Try intelligent segmentation first using silence detection
        try:
            non_silent_regions = self._detect_silence_boundaries(audio, sampling_rate)
            
            segment_id = 0
            for region_start, region_end in non_silent_regions:
                region_audio = audio[region_start:region_end]
                region_duration = len(region_audio) / sampling_rate
                
                # If region is shorter than effective minimum duration, skip
                if region_duration < effective_min_duration:
                    continue
                
                # If region fits in one segment, use it
                if region_duration <= self.config.max_segment_duration:
                    segments.append({
                        "audio": region_audio,
                        "labels": label_id,
                        "label_name": self.id_to_label.get(label_id, f"unknown_{label_id}"),
                        "sampling_rate": sampling_rate,
                        "segment_id": segment_id,
                        "source_file": audio_path,
                        "segment_start": region_start / sampling_rate,
                        "segment_duration": region_duration
                    })
                    segment_id += 1
                else:
                    # Split long regions into overlapping chunks
                    step_samples = target_samples - overlap_samples
                    for start_offset in range(0, len(region_audio) - effective_min_samples + 1, step_samples):
                        end_offset = min(start_offset + target_samples, len(region_audio))
                        
                        # Ensure effective minimum segment length
                        if end_offset - start_offset < effective_min_samples:
                            continue
                        
                        segment_audio = region_audio[start_offset:end_offset]
                        segment_duration = len(segment_audio) / sampling_rate
                        
                        segments.append({
                            "audio": segment_audio,
                            "labels": label_id,
                            "label_name": self.id_to_label.get(label_id, f"unknown_{label_id}"),
                            "sampling_rate": sampling_rate,
                            "segment_id": segment_id,
                            "source_file": audio_path,
                            "segment_start": (region_start + start_offset) / sampling_rate,
                            "segment_duration": segment_duration
                        })
                        segment_id += 1
        
        except Exception as e:
            print(f"Warning: Intelligent segmentation failed for {audio_path}: {e}")
            print("Falling back to simple chunking...")
            
            # Fallback to simple overlapping windows
            segments = []
            step_samples = target_samples - overlap_samples
            segment_id = 0
            
            for start_idx in range(0, len(audio) - effective_min_samples + 1, step_samples):
                end_idx = min(start_idx + target_samples, len(audio))
                
                # Ensure effective minimum segment length
                if end_idx - start_idx < effective_min_samples:
                    continue
                
                segment_audio = audio[start_idx:end_idx]
                segment_duration = len(segment_audio) / sampling_rate
                
                segments.append({
                    "audio": segment_audio,
                    "labels": label_id,
                    "label_name": self.id_to_label.get(label_id, f"unknown_{label_id}"),
                    "sampling_rate": sampling_rate,
                    "segment_id": segment_id,
                    "source_file": audio_path,
                    "segment_start": start_idx / sampling_rate,
                    "segment_duration": segment_duration
                })
                segment_id += 1
        
        return segments if segments else [{
            "audio": audio,
            "labels": label_id,
            "label_name": self.id_to_label.get(label_id, f"unknown_{label_id}"),
            "sampling_rate": sampling_rate,
            "segment_id": 0,
            "source_file": audio_path,
            "segment_start": 0.0,
            "segment_duration": duration
        }]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset with AST feature extraction."""
        # Get the raw data
        sample = self.data[idx]
        
        # Extract AST features using the feature extractor
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
            "segment_duration": sample.get("segment_duration", len(audio) / self.config.sampling_rate)
        }
    
    def _process_sample(self, sample: Any, correct_labels_map: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """Process a single sample into the format expected by models."""
        # Extract audio data - now audio is not pre-decoded due to decode=False
        audio_info = sample["audio"]
        
        if isinstance(audio_info, dict):
            if "array" in audio_info and audio_info["array"] is not None:
                # Audio is already loaded (shouldn't happen with decode=False, but handle it)
                audio = audio_info["array"]
                sampling_rate = audio_info["sampling_rate"]
                audio_path = audio_info.get("path", "unknown")
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
                    audio_path = "bytes_data"
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
                    audio_path = audio_info
                    audio, sampling_rate = sf.read(audio_info)
                else:
                    # It's bytes
                    import io
                    audio, sampling_rate = sf.read(io.BytesIO(audio_info))
                    audio_path = "bytes_data"
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
        
        # Check minimum length for AST feature extractor
        # AST requires at least 400 samples for its window size
        min_length = 400  # Minimum samples needed for AST
        if len(audio) < min_length:
            # Pad with zeros if too short
            padding_needed = min_length - len(audio)
            audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)
            print(f"Warning: Audio too short ({len(audio) - padding_needed} samples), padded to {len(audio)} samples")
        
        # Also ensure minimum duration for Wav2Vec2 compatibility (at least 0.5 seconds)
        # Wav2Vec2 models require sufficient sequence length for masking
        min_duration_samples = int(0.5 * self.config.sampling_rate)  # 0.5 seconds minimum
        if len(audio) < min_duration_samples:
            padding_needed = min_duration_samples - len(audio)
            audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)
        
        # Additional check: Ensure at least 1 second for very reliable Wav2Vec2 processing
        # This helps avoid the mask_length > sequence_length error
        wav2vec2_min_samples = int(1.0 * self.config.sampling_rate)  # 1 second minimum for Wav2Vec2
        if len(audio) < wav2vec2_min_samples:
            padding_needed = wav2vec2_min_samples - len(audio)
            audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)
        
        # Resample if needed
        if sampling_rate != self.config.sampling_rate:
            audio = self.resample_audio(audio, sampling_rate, self.config.sampling_rate)
        
        # Apply cleaning if enabled
        audio = self.clean_audio(audio, self.config.sampling_rate)
        
        # Final ensure float32 type after all processing
        audio = audio.astype(np.float32)
        
        # Final length check after all processing
        if len(audio) < min_length:
            padding_needed = min_length - len(audio)
            audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)
        
        # IMPORTANT: Return raw audio, not pre-processed features
        # The model will handle feature extraction during training
        
        # Get label information
        if "label" in sample:
            label_id = sample["label"]
        elif "labels" in sample:
            label_id = sample["labels"]
        elif "sex" in sample:
            # Map sex to label_id
            sex_val = sample["sex"]
            if sex_val == "female" or sex_val == 0:
                label_id = 0
            elif sex_val == "male" or sex_val == 1:
                label_id = 1
            else:
                raise ValueError(f"Unknown sex value: {sex_val}")
        else:
            raise ValueError(f"No label found in sample. Available keys: {list(sample.keys())}")
        
        # Filter out invalid labels
        if label_id >= 2:
            raise ValueError(f"Invalid label {label_id}")
        
        label_name = self.id_to_label.get(label_id, f"unknown_{label_id}")
        
        # Apply segmentation if enabled
        if self.config.enable_segmentation:
            return self._segment_audio(audio, self.config.sampling_rate, label_id, audio_path)
        else:
            # Return single sample
            return [{
                "audio": audio,  # Raw audio array
                "labels": label_id,
                "label_name": label_name,
                "sampling_rate": self.config.sampling_rate,
                "segment_id": 0,
                "source_file": audio_path,
                "segment_start": 0.0,
                "segment_duration": len(audio) / self.config.sampling_rate
            }]


def create_simple_barkopedia_gender_dataset(
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
        enable_segmentation: Whether to enable automatic audio segmentation
        segment_duration: Target duration for each segment (0.3-5.0 seconds)
        segment_overlap: Overlap between segments in seconds
        energy_threshold: Energy threshold for silence detection (0.001-0.1)
    
    Returns:
        Dictionary with 'train', 'validation', and 'test' dataset instances
    """
    
    # Create configuration
    config = SimpleDatasetConfig(
        dataset_name="barkopedia_gender",
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
    
    # Create train, validation, and test datasets
    train_dataset = SimpleBarkopediaGenderDataset(config, split="train")
    train_dataset.load_data()
    
    validation_dataset = SimpleBarkopediaGenderDataset(config, split="validation")
    validation_dataset.load_data()
    
    test_dataset = SimpleBarkopediaGenderDataset(config, split="test")
    test_dataset.load_data()
    
    return {
        "train": train_dataset,
        "validation": validation_dataset,
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
        max_duration=None  # Allow long files for segmentation testing
    )
    
    train_dataset = splits["train"]
    test_dataset = splits["test"]
    
    print(f"   ✓ Train dataset: {len(train_dataset)} samples")
    print(f"   ✓ Test dataset: {len(test_dataset)} samples")
    
    # Test segmentation
    print("\n2. Testing segmentation feature...")
    segmented_splits = create_simple_barkopedia_gender_dataset(
        hf_token=hf_token,
        apply_cleaning=True,
        enable_segmentation=True,
        segment_duration=2.0,
        segment_overlap=0.1,
        energy_threshold=0.01
    )
    
    segmented_train = segmented_splits["train"]
    segmented_test = segmented_splits["test"]
    
    print(f"   ✓ Segmented Train dataset: {len(segmented_train)} segments")
    print(f"   ✓ Segmented Test dataset: {len(segmented_test)} segments")
    print(f"   ✓ Segmentation increased samples by: {len(segmented_train)/len(train_dataset):.2f}x")
    
    # Test sample access
    print("\n3. Testing sample access...")
    train_sample = train_dataset[0]
    segmented_sample = segmented_train[0]
    
    print(f"   ✓ Original sample keys: {list(train_sample.keys())}")
    print(f"   ✓ Segmented sample keys: {list(segmented_sample.keys())}")
    print(f"   ✓ Input values shape: {train_sample['input_values'].shape}")
    print(f"   ✓ Label: {train_sample['labels']} ({train_sample['label_name']})")
    print(f"   ✓ Sampling rate: {train_sample['sampling_rate']}")
    print(f"   ✓ Audio shape: {train_sample['audio'].shape}")
    
    # Check segmented sample metadata
    if 'segment_id' in segmented_sample:
        print(f"   ✓ Segment ID: {segmented_sample['segment_id']}")
        print(f"   ✓ Source file: {segmented_sample.get('source_file', 'N/A')}")
        print(f"   ✓ Segment start: {segmented_sample.get('segment_start', 0):.2f}s")
        print(f"   ✓ Segment duration: {segmented_sample.get('segment_duration', 0):.2f}s")
    
    # Test class distribution
    print("\n4. Class distribution:")
    train_dist = train_dataset.get_class_distribution()
    for label, count in train_dist.items():
        print(f"   Original {label}: {count} samples")
    
    segmented_train_dist = segmented_train.get_class_distribution()
    for label, count in segmented_train_dist.items():
        print(f"   Segmented {label}: {count} segments")
    
    # Test DataLoader
    print("\n5. Testing DataLoader...")
    dataloader = train_dataset.get_dataloader(batch_size=4, shuffle=False)
    batch = next(iter(dataloader))
    
    print(f"   ✓ Batch input_values shape: {batch['input_values'].shape}")
    print(f"   ✓ Batch labels shape: {batch['labels'].shape}")
    print(f"   ✓ Labels in batch: {batch['labels']}")
    
    # Test segmented DataLoader
    print("\n6. Testing segmented DataLoader...")
    segmented_dataloader = segmented_train.get_dataloader(batch_size=4, shuffle=False)
    segmented_batch = next(iter(segmented_dataloader))
    
    print(f"   ✓ Segmented batch input_values shape: {segmented_batch['input_values'].shape}")
    print(f"   ✓ Segmented batch labels shape: {segmented_batch['labels'].shape}")
    print(f"   ✓ Segmented labels in batch: {segmented_batch['labels']}")
    
    print("\n=== All tests passed! ===")
