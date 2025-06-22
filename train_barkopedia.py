# Fine-tune MIT/ast-finetuned-audioset-10-10-0.4593 on Barkopedia
import os
from datasets import load_dataset, load_from_disk
from transformers import ASTFeatureExtractor, ASTForAudioClassification, TrainingArguments, Trainer
import torch

# Load Hugging Face token from hf_token.py if available
hf_token_path = os.path.join(os.path.dirname(__file__), 'hf_token.py')
if os.path.exists(hf_token_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('hf_token', hf_token_path)
    hf_token_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf_token_module)
    HF_TOKEN = getattr(hf_token_module, 'HF_TOKEN', None)
else:
    HF_TOKEN = os.environ.get("HF_TOKEN")

# Load dataset
local_dir = "./barkopedia_dataset"
print("Loading dataset from Hugging Face Hub...")
ds = load_dataset(
    "ArlingtonCL2/Barkopedia_DOG_AGE_GROUP_CLASSIFICATION_DATASET",
    token=HF_TOKEN,
    cache_dir=local_dir,
)
print("Dataset loaded successfully.")

# Use the first available split as train, second as test (customize as needed)
splits = list(ds.keys())
train_ds = ds[splits[0]]
test_ds = ds[splits[1]] if len(splits) > 1 else None
# print available splits and their sizes
print("Available splits:", ds.keys())
for split in ds.keys():
    print(f"{split}: {len(ds[split])} samples")

# print columns and labels
print("Columns in train dataset:", train_ds.column_names)
print("Labels:", set(train_ds["label"]))


# Load feature extractor and model
feature_extractor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

# Get number of unique labels
num_labels = len(set(train_ds["label"]))

model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# Set which GPU to use
GPU_NUM = 1  # Change this to the desired GPU index
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NUM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess(batch):
    audio = batch["audio"]
    inputs = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    )

    batch["input_values"] = inputs.input_values[0]
    batch["labels"] = batch["label"]
    return batch

# Preprocessing cache paths
train_cache = "./train_preprocessed"
test_cache = "./test_preprocessed"

if os.path.exists(train_cache) and os.path.exists(test_cache):
    print("Loading preprocessed datasets from disk cache...")
    train_ds = load_from_disk(train_cache)
    test_ds = load_from_disk(test_cache)
else:
    print("Preprocessing datasets (this may take a while)...")
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    test_ds = test_ds.map(preprocess, remove_columns=test_ds.column_names)
    print("Saving preprocessed datasets to disk cache...")
    train_ds.save_to_disk(train_cache)
    test_ds.save_to_disk(test_cache)

train_ds.set_format(type="torch", columns=["input_values", "labels"])
test_ds.set_format(type="torch", columns=["input_values", "labels"])

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds if test_ds else None,
)

trainer.train()
