import os
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# Set device
GPU_NUM = 2  # Change if needed
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NUM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and feature extractor
model_dir = "./barkopedia_finetuned_model"
model = ASTForAudioClassification.from_pretrained(model_dir).to(device)
feature_extractor = ASTFeatureExtractor.from_pretrained(model_dir)

# Load test set (from cache if available)
test_cache = "./test_preprocessed"
if os.path.exists(test_cache):
    test_ds = load_from_disk(test_cache)
else:
    # Fallback: load from hub (slow)
    from datasets import load_dataset
    ds = load_dataset("ArlingtonCL2/Barkopedia_DOG_AGE_GROUP_CLASSIFICATION_DATASET")
    splits = list(ds.keys())
    test_ds = ds[splits[1]] if len(splits) > 1 else ds[splits[0]]

# Use explicit id2label mapping
id2label_dict = {
    0: "adolescent",
    1: "adult",
    2: "juvenile",
    3: "puppy",
    4: "senior"
}
id2label = lambda x: id2label_dict.get(x, str(x))
# id2label = lambda x: x # for debugging

results = []
model.eval()
with torch.no_grad():
    for example in test_ds:
        audio = example["audio"]
        inputs = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).to(device)
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
        pred_label = id2label(pred_id)
        audio_id = os.path.splitext(os.path.basename(audio["path"]))[0]
        results.append({
            "audio_id": audio_id,
            "pred_dog_age_group": pred_label
        })

# Save to CSV
out_csv = "test_predictions.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"Predictions saved to {out_csv}")
