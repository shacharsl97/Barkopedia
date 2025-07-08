# Fine-tune MIT/ast-finetuned-audioset-10-10-0.4593 on Barkopedia
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
from transformers import EarlyStoppingCallback
from dog_age_dataset import create_dog_age_dataset

# Age order and distance matrix
age_order = {4: 4, 3: 3, 0: 0, 2: 2, 1: 1}
n_classes = 5
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

# {'learning_rate': 2.5e-05, 'weight_decay': 0.01, 'hidden_dropout_prob': 0.35, 'attention_probs_dropout_prob': 0.35, 'clean_audio': '', 'loss': 'soft_cross', 'num_train_epochs': 6}
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=2.5e-05)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--hidden_dropout_prob', type=float, default=0.35)
parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.35)
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--data_mode', choices=['original', 'augmented'], default='original', help='Which training data to use')
parser.add_argument('--metrics_out', type=str, default=None, help='If set, write final metrics to this file')
parser.add_argument('--gpu_num', type=int, default=0, help='GPU number to use (default: 0)')
parser.add_argument('--test_20', action='store_true', help='Run with just 20 steps for debugging')
parser.add_argument('--loss', choices=['old','soft_cross'], default='soft_cross')
parser.add_argument('--clean_audio', action='store_true', help='If set, clean audio with librosa in preprocess')
parser.add_argument('--backbone', choices=['ast', 'huber', 'wav2vec2'], default='ast', help='Model backbone: ast, huber, or wav2vec2')
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

# Load dataset using new dog_age_dataset loader
local_dir = "./barkopedia_dataset"
print("Loading dataset from Hugging Face Hub...")
datasets = create_dog_age_dataset(
    cache_dir=local_dir,
    hf_token=HF_TOKEN,
    apply_cleaning=args.clean_audio,
    sampling_rate=16000,
    backbone=args.backbone,
)
train_dataset = datasets["train"]
# Use validation split if available, else fallback to test
if "validation" in datasets:
    eval_dataset = datasets["validation"]
else:
    eval_dataset = datasets["test"]

# DataLoader setup (like train_individual_dog_recognition)
from torch.utils.data import DataLoader
train_loader = train_dataset.get_dataloader(batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
eval_loader = eval_dataset.get_dataloader(batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Set up for Trainer
# Note: We are not using Trainer class here, but keeping this section for potential future use
train_ds = train_dataset
eval_ds = eval_dataset

print(f"Initial train_ds: {len(train_ds)} samples")
print(f"Initial eval_ds: {len(eval_ds)} samples")

# Load feature extractor and model based on backbone
if args.backbone == 'ast':
    from transformers import ASTFeatureExtractor, ASTForAudioClassification, ASTConfig
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
    config = ASTConfig.from_pretrained(
        model_name,
        num_labels=5,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob
    )
    ModelClass = ASTForAudioClassification
elif args.backbone == 'huber':
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2Config
    model_name = "facebook/hubert-large-ls960-ft"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=5,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob
    )
    ModelClass = Wav2Vec2ForSequenceClassification
elif args.backbone == 'wav2vec2':
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2Config
    model_name = "facebook/wav2vec2-large-960h"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=5,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob
    )
    ModelClass = Wav2Vec2ForSequenceClassification
else:
    raise ValueError(f"Unknown backbone: {args.backbone}")

MAX_AUDIO_LEN = 50000  # ~3.1 seconds at 16kHz

def preprocess(batch):
    audio_array = batch["audio"]
    sr = batch["audio"]["sampling_rate"]
    import librosa
    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    audio_array, _ = librosa.effects.trim(audio_array)
    audio_array = librosa.util.normalize(audio_array)
    sr = 16000

    if len(audio_array) > MAX_AUDIO_LEN and args.backbone == 'wav2vec2':
            audio_array = audio_array[:MAX_AUDIO_LEN]

    inputs = feature_extractor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt"
        )
    batch["input_values"] = inputs.input_values[0]
    return batch

# Model selection
if args.loss == 'old':
    model = ModelClass.from_pretrained(
        model_name,
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
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
    elif args.backbone == 'huber' or args.backbone == 'wav2vec2':
        class OrdinalWav2Vec2Classifier(Wav2Vec2ForSequenceClassification):
            def forward(self, input_values=None, labels=None, **kwargs):
                print(input_values.shape)  # should be [batch_size, sequence_length]
                kwargs.pop('num_items_in_batch', None)
                outputs = super().forward(input_values=input_values, labels=None, **kwargs)
                logits = outputs.logits
                loss = soft_cross_entropy_with_distance(logits, labels, distance_matrix) if labels is not None else None
                from transformers.modeling_outputs import SequenceClassifierOutput
                return SequenceClassifierOutput(loss=loss, logits=logits)
        model = OrdinalWav2Vec2Classifier.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
    else:
        model = ModelClass.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class_positions = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32).to(device)

# Loss: Use mean squared error loss for age regression/classification
class AgeMSELoss(torch.nn.Module):
    def __init__(self, class_positions):
        super().__init__()
        self.class_positions = class_positions
    def forward(self, logits, labels):
        probs = torch.nn.functional.softmax(logits, dim=1)
        expected = torch.sum(probs * self.class_positions, dim=1)
        true = self.class_positions[labels]
        return torch.mean((expected - true) ** 2)

# Training loop (like IndividualDogTrainer, but simpler)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = AgeMSELoss(class_positions)

for epoch in range(1, args.num_train_epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(input_values=batch['input_values'])
        logits = outputs.logits
        loss = criterion(logits, batch['labels'])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(input_values=batch['input_values'])
            logits = outputs.logits
            loss = criterion(logits, batch['labels'])
            val_loss += loss.item()
    avg_val_loss = val_loss / len(eval_loader)
    print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")

# Save the best model and feature extractor
save_dir = "./barkopedia_finetuned_model"
print(f"Saving best model and feature extractor to {save_dir} ...")
torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
feature_extractor.save_pretrained(save_dir)
print("Best model and feature extractor saved.")

# Evaluate best model on eval set and print best metrics
print("Evaluating best model on eval set...")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in eval_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(input_values=batch['input_values'])
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Eval Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Write final metrics to file if requested
if args.metrics_out:
    print(f"Writing final metrics to {args.metrics_out}")
    with open(args.metrics_out, 'w') as f:
        json.dump({
            'eval_accuracy': accuracy,
            'eval_f1': f1,
        }, f)
