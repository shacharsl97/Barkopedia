import librosa
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Load dataset
ds = load_dataset("ArlingtonCL2/Barkopedia_DOG_AGE_GROUP_CLASSIFICATION_DATASET", cache_dir="./barkopedia_dataset")
splits = list(ds.keys())
train_ds = ds[splits[0]]
test_ds = ds[splits[1]]
train_ds, test_ds = train_ds, test_ds

def extract_mfcc(sample, n_mfcc=13):
    y = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # shape: (n_mfcc,)

# Extract features
X_train = np.array([extract_mfcc(sample) for sample in tqdm(train_ds, desc="Extracting train MFCCs")])
y_train = np.array(train_ds["label"])

X_test = np.array([extract_mfcc(sample) for sample in tqdm(test_ds, desc="Extracting test MFCCs")])
y_test = np.array(test_ds["label"])

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"MFCC + Logistic Regression Accuracy: {acc:.4f}")
print(f"MFCC + Logistic Regression F1: {f1:.4f}")
