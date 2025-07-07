# Barkopedia Datasets - Module Summary

## Module Structure

The datasets have been successfully reorganized into the `barkopedia_datasets` module to avoid conflicts with the HuggingFace `datasets` library.

### Available Datasets

#### Age Group Classification
- **Simple**: `SimpleBarkopediaAgeDataset`
- **Full**: `BarkopediaAgeGroupDataset`
- **Labels**: adolescent, adult, juvenile, puppy, senior

#### Gender Classification  
- **Simple**: `SimpleBarkopediaGenderDataset`
- **Full**: `BarkopediaGenderDataset`
- **Labels**: female, male

### Usage Examples

#### Simple Usage (Recommended)
```python
from barkopedia_datasets import create_simple_barkopedia_gender_dataset

# Create gender classification dataset
splits = create_simple_barkopedia_gender_dataset(
    hf_token=your_token,
    apply_cleaning=True,
    max_duration=5.0
)

train_dataset = splits['train']
test_dataset = splits['test']

# Use with PyTorch DataLoader
dataloader = train_dataset.get_dataloader(batch_size=32, shuffle=True)
```

#### Age Classification
```python
from barkopedia_datasets import create_simple_barkopedia_dataset

splits = create_simple_barkopedia_dataset(
    hf_token=your_token,
    apply_cleaning=True
)
```

### Features

1. **Simple Interface**: Just holds data and works with PyTorch DataLoader
2. **Cleaning Option**: Optional audio cleaning in the constructor
3. **PyTorch Compatible**: Direct DataLoader support
4. **Both Tasks**: Age group and gender classification
5. **HuggingFace Integration**: Automatic download from HuggingFace Hub

### Module Structure
```
barkopedia_datasets/
├── __init__.py
├── simple_dataset_interface.py
├── simple_barkopedia_age_dataset.py
├── simple_barkopedia_gender_dataset.py
├── dataset_interface.py (full interface)
├── barkopedia_age_dataset.py (full implementation)
└── barkopedia_gender_dataset.py (full implementation)
```

The module successfully avoids naming conflicts and provides clean, simple access to both age group and gender classification datasets.
