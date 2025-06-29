# Fine-tune audio backbone models on Barkopedia Dog Sex Classification
import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import argparse
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
import json

# Age order and distance matrix (not used for sex, but kept for compatibility)
age_order = {0: 0, 1: 1}
n_classes = 2
distance_matrix = torch.zeros((n_classes, n_classes))
for i in range(n_classes):
    for j in range(n_classes):
        distance_matrix[i, j] = abs(age_order[i] - age_order[j])
print("Distance matrix:", distance_matrix)

def soft_cross_entropy_with_distance(logits, true_labels, distance_matrix, temperature=0.8):
    device = logits.device
    num_classes = logits.size(1)
    dist_targets = distance_matrix.to(device)[true_labels]  # [B, C]
    sim = -dist_targets / temperature  # [B, C]
    soft_targets = torch.nn.functional.softmax(sim, dim=1)  # [B, C]
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)  # [B, C]
    loss = torch.sum(-soft_targets * log_probs, dim=1)  # [B]
    return torch.mean(loss)

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=1.5e-05)
parser.add_argument('--weight_decay', type=float, default=0.02)
parser.add_argument('--hidden_dropout_prob', type=float, default=0.3)
parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.3)
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--data_mode', choices=['original', 'augmented'], default='original', help='Which training data to use')
parser.add_argument('--metrics_out', type=str, default=None, help='If set, write final metrics to this file')
parser.add_argument('--gpu_num', type=int, default=0, help='GPU number to use (default: 0)')
parser.add_argument('--test_20', action='store_true', help='Run with just 20 steps for debugging')
parser.add_argument('--loss', choices=['old','soft_cross'], default='old')
parser.add_argument('--clean_audio', action='store_true', help='If set, clean audio with librosa in preprocess')
parser.add_argument('--backbone', choices=['ast', 'beats', 'wav2vec2'], default='ast', help='Model backbone: ast, beats, or wav2vec2')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

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

# Load dataset (Dog Sex Classification)
local_dir = "./barkopedia_dataset"
print("Loading dataset from Hugging Face Hub...")
ds = load_dataset(
    "ArlingtonCL2/Barkopedia_Dog_Sex_Classification_Dataset",
    token=HF_TOKEN,
    cache_dir=local_dir,
)
print("Dataset loaded successfully.")

# Use the first available split as train, second as test (customize as needed)
splits = list(ds.keys())
train_ds = ds[splits[0]]
test_ds = ds[splits[1]]
print(f"Initial train_ds: {len(train_ds)} samples")
print(f"Initial test_ds: {len(test_ds)} samples")
print("Available splits:", ds.keys())
for split in ds.keys():
    print(f"{split}: {len(ds[split])} samples")

print("Columns in train dataset:", train_ds.column_names)
print("Labels:", set(train_ds["label"]))

# Determine number of classes dynamically from dataset
class_names = sorted(list(set([item['label'] for item in train_ds])))
num_labels = len(class_names)
print("Class names:", class_names)
print("Number of labels:", num_labels)

# Load feature extractor and model based on backbone
if args.backbone == 'ast':
    from transformers import ASTFeatureExtractor, ASTForAudioClassification, ASTConfig
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    config = ASTConfig.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=num_labels,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob
    )
    ModelClass = ASTForAudioClassification
elif args.backbone == 'beats':
    from transformers.models import BEATSFeatureExtractor, BEATSForAudioClassification, BEATSConfig
    feature_extractor = BEATSFeatureExtractor.from_pretrained(
        "microsoft/beats-finetuned-audioset-epoch-16"
    )
    config = BEATSConfig.from_pretrained(
        "microsoft/beats-finetuned-audioset-epoch-16",
        num_labels=num_labels,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob
    )
    ModelClass = BEATSForAudioClassification
elif args.backbone == 'wav2vec2':
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2Config
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    config = Wav2Vec2Config.from_pretrained(
        "facebook/wav2vec2-base-960h",
        num_labels=num_labels,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob
    )
    ModelClass = Wav2Vec2ForSequenceClassification
else:
    raise ValueError(f"Unknown backbone: {args.backbone}")

def preprocess(batch):
    if "audio" in batch and isinstance(batch["audio"], dict) and "array" in batch["audio"]:
        audio_array = batch["audio"]["array"]
        sr = batch["audio"]["sampling_rate"]
        if args.clean_audio:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            audio_array, _ = librosa.effects.trim(audio_array)
            audio_array = librosa.util.normalize(audio_array)
            sr = 16000
        # Always ensure 1D waveform for all backbones
        audio_array = np.asarray(audio_array).flatten().astype(np.float32)
        inputs = feature_extractor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt"
        )
        batch["input_values"] = inputs.input_values[0]
    batch["labels"] = batch["label"]
    return batch

# Preprocessing cache paths (separate for original and augmented)
if args.data_mode == 'original':
    train_cache = "./train_preprocessed"
    test_cache = "./test_preprocessed"
else:
    train_cache = "./train_preprocessed_aug"
    test_cache = "./test_preprocessed_aug"

if args.data_mode == 'original':
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
elif args.data_mode == 'augmented':
    if os.path.exists(train_cache):
        print(f"Loading concatenated original+augmented training set from cache: {train_cache}")
        train_ds = load_from_disk(train_cache)
    else:
        print("Preprocessing original data for input_values...")
        train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
        train_ds.set_format(type="torch", columns=["input_values", "labels"])
        print("Preparing augmented data and attaching input_values...")
        meta = pd.read_csv('augmented_train/augmented_metadata.csv', usecols=["audio_path", "label", "aug_type", "base_id"])
        input_values_map = np.load('augmented_train/input_values_map.npy', allow_pickle=True).item()
        meta["input_values"] = meta["base_id"].map(lambda k: np.array(input_values_map[k], dtype=np.float32).astype(np.float32).tolist())
        meta = meta.rename(columns={"label": "labels"})
        features = Features({
            "audio_path": Value("string"),
            "labels": Value("int64"),
            "aug_type": Value("string"),
            "base_id": Value("string"),
            "input_values": Sequence(feature=Sequence(feature=Value("float32")))
        })
        aug_ds = Dataset.from_pandas(meta, features=features)
        aug_ds.set_format(type="torch", columns=["input_values", "labels"])
        print("Concatenating original and augmented datasets...")
        train_ds = concatenate_datasets([train_ds, aug_ds])
        print(f"Saving concatenated dataset to cache: {train_cache}")
        train_ds.save_to_disk(train_cache)
    # Handle eval set (test_ds)
    if os.path.exists(test_cache):
        print(f"Loading eval set from cache: {test_cache}")
        test_ds = load_from_disk(test_cache)
    else:
        print("Preprocessing eval set (test_ds) for input_values...")
        test_ds = test_ds.map(preprocess, remove_columns=test_ds.column_names)
        test_ds.set_format(type="torch", columns=["input_values", "labels"])
        print(f"Saving eval set to cache: {test_cache}")
        test_ds.save_to_disk(test_cache)

# Model selection
if args.loss == 'old':
    model = ModelClass.from_pretrained(
        feature_extractor.name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
    )
else:
    if args.backbone == 'ast':
        class OrdinalASTClassifier(ASTForAudioClassification):
            def forward(self, input_values=None, labels=None, **kwargs):
                kwargs.pop('num_items_in_batch', None)
                outputs = super().forward(input_values=input_values, labels=None, **kwargs)
                logits = outputs.logits
                loss = None
                if labels is not None:
                    if args.loss == 'soft_cross':
                        loss = soft_cross_entropy_with_distance(logits, labels, distance_matrix)
                from transformers.modeling_outputs import SequenceClassifierOutput
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
                    attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
                )
        model = OrdinalASTClassifier.from_pretrained(
            feature_extractor.name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
        )
    else:
        model = ModelClass.from_pretrained(
            feature_extractor.name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Move class_positions tensors to device (not used for sex, but kept for compatibility)
class_positions_order_aware = torch.tensor([0.0, 1.0], dtype=torch.float32).to(device)
class_positions_order_agnostic = torch.tensor([1.0, 0.0], dtype=torch.float32).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=args.num_train_epochs,
    weight_decay=args.weight_decay,  # Use from args
    logging_dir="./logs",
    logging_steps=10,
    dataloader_num_workers=4,
    max_steps=20 if args.test_20 else -1
)

def mean_squared_error_sex_distance(preds, labels):
    mse_arr = []
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)
    for class_positions in [class_positions_order_aware, class_positions_order_agnostic]:
        class_pos_np = class_positions.cpu().numpy() if hasattr(class_positions, 'cpu') else np.array(class_positions)
        expected_pos = np.sum(preds * class_pos_np, axis=1)
        true_pos = class_pos_np[labels_np]
        mse = np.mean((expected_pos - true_pos) ** 2)
        mse_arr.append(mse)
    return mse_arr

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    mse = mean_squared_error_sex_distance(logits, labels)
    return {"accuracy": acc, "f1": f1, "mse_order_aware": mse[0], "mse_order_agnostic": mse[1]}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

if not args.metrics_out:
    # Save the fine-tuned model and feature extractor
    save_dir = "./barkopedia_finetuned_model"
    print(f"Saving model and feature extractor to {save_dir} ...")
    model.save_pretrained(save_dir)
    feature_extractor.save_pretrained(save_dir)
    print("Model and feature extractor saved.")

# Write final metrics to file if requested
if args.metrics_out:
    print(f"Writing final metrics to {args.metrics_out}")
    # Get last train and eval metrics from trainer.state.log_history
    train_loss = eval_loss = train_acc = eval_acc = train_f1 = eval_f1 = None
    for entry in reversed(trainer.state.log_history):
        # Train loss: last entry with 'loss' and 'epoch' and not 'eval_loss'
        if train_loss is None and 'loss' in entry and 'epoch' in entry and 'eval_loss' not in entry:
            train_loss = entry['loss']
        # Eval loss: last entry with 'eval_loss'
        if eval_loss is None and 'eval_loss' in entry:
            eval_loss = entry['eval_loss']
        # Eval accuracy: last entry with 'eval_accuracy'
        if eval_acc is None and 'eval_accuracy' in entry:
            eval_acc = entry['eval_accuracy']
        # Eval f1: last entry with 'eval_f1'
        if eval_f1 is None and 'eval_f1' in entry:
            eval_f1 = entry['eval_f1']
        # Stop early if all found
        if all(x is not None for x in [train_loss, eval_loss, eval_acc, eval_f1]):
            break
    # Optionally, run evaluation on train set for train_acc/train_f1
    train_acc = train_f1 = None
    if hasattr(trainer, 'compute_metrics') and hasattr(trainer, 'evaluate'):
        train_metrics = trainer.evaluate(train_ds, metric_key_prefix='train')
        train_acc = train_metrics.get('train_accuracy')
        train_f1 = train_metrics.get('train_f1')
    with open(args.metrics_out, 'w') as f:
        json.dump({
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'train_acc': train_acc,
            'eval_acc': eval_acc,
            'train_f1': train_f1,
            'eval_f1': eval_f1,
        }, f)
