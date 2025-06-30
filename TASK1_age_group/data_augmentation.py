import os
import pandas as pd
from datasets import load_dataset
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from transformers import ASTFeatureExtractor

# Directory to save augmented files and metadata
AUG_DIR = "augmented_train"
META_CSV = os.path.join(AUG_DIR, "augmented_metadata.csv")

os.makedirs(AUG_DIR, exist_ok=True)

# Load dataset
local_dir = "./barkopedia_dataset"
ds = load_dataset(
    "ArlingtonCL2/Barkopedia_DOG_AGE_GROUP_CLASSIFICATION_DATASET",
    cache_dir=local_dir,
)
splits = list(ds.keys())
train_ds = ds[splits[0]]

# Load feature extractor
feature_extractor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

rows = []
for idx, example in tqdm(enumerate(train_ds), total=len(train_ds)):
    audio = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    label = example["label"]
    base_id = os.path.splitext(os.path.basename(example["audio"].get("path", f"bark_{idx}.wav")))[0]
    # Save original
    orig_path = os.path.join(AUG_DIR, f"{base_id}_orig.wav")
    sf.write(orig_path, audio, sr)
    # Compute and save input_values for original
    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="np")
    orig_input_values = inputs.input_values[0]
    rows.append({
        "audio_path": orig_path,
        "label": label,
        "aug_type": "original",
        "base_id": base_id,
        "input_values": orig_input_values.tolist()
    })
    # Augmentations
    augments = {
        "noise": audio + 0.005 * np.random.randn(*audio.shape),
        "pitch_shift": librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=2),
        "time_stretch": librosa.effects.time_stretch(audio, rate=0.8),
    }
    for aug_name, aug_audio in augments.items():
        aug_path = os.path.join(AUG_DIR, f"{base_id}_{aug_name}.wav")
        sf.write(aug_path, aug_audio, sr)
        # Compute and save input_values for augmented audio
        aug_inputs = feature_extractor(aug_audio, sampling_rate=sr, return_tensors="np")
        aug_input_values = aug_inputs.input_values[0]
        rows.append({
            "audio_path": aug_path,
            "label": label,
            "aug_type": aug_name,
            "base_id": base_id,
            "input_values": aug_input_values.tolist()
        })

# Save metadata with input_values as a column
pd.DataFrame(rows).to_csv(META_CSV, index=False)
print(f"Augmented data and metadata (with input_values) saved in {AUG_DIR}")
