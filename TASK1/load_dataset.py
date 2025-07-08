# Add your Hugging Face token here or set it as an environment variable
HF_TOKEN = None  # Or: 'hf_xxx...'

from datasets import load_dataset
import os

hf_token_path = os.path.join(os.path.dirname(__file__), 'hf_token.py')
if os.path.exists(hf_token_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('hf_token', hf_token_path)
    hf_token_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf_token_module)
    HF_TOKEN = getattr(hf_token_module, 'HF_TOKEN', None)
else:
    HF_TOKEN = os.environ.get("HF_TOKEN")

# Always download from Hugging Face Hub if local_dir does not exist
local_dir = "./barkopedia_dataset"

# Load the dataset without specifying splits
ds = load_dataset(
    "ArlingtonCL2/Barkopedia_DOG_AGE_GROUP_CLASSIFICATION_DATASET",
    token=HF_TOKEN,
    cache_dir=local_dir,

)

# Print available splits and their sizes
print("Available splits:", ds.keys())
for split in ds.keys():
    print(f"{split}: {len(ds[split])} samples")
