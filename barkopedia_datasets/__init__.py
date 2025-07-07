# Barkopedia Datasets Package
from .dataset_interface import BarkopediaDataset, DatasetConfig, AudioDatasetMixin
from .barkopedia_age_dataset import BarkopediaAgeGroupDataset, create_barkopedia_age_dataset
from .barkopedia_gender_dataset import BarkopediaGenderDataset, create_barkopedia_gender_dataset

# Simple interfaces
from .simple_dataset_interface import SimpleBarkopediaDataset, SimpleDatasetConfig
from .simple_barkopedia_age_dataset import SimpleBarkopediaAgeDataset, create_simple_barkopedia_dataset
from .simple_barkopedia_gender_dataset import SimpleBarkopediaGenderDataset, create_simple_barkopedia_gender_dataset

__all__ = [
    'BarkopediaDataset', 
    'DatasetConfig', 
    'AudioDatasetMixin',
    'BarkopediaAgeGroupDataset', 
    'create_barkopedia_age_dataset',
    'BarkopediaGenderDataset', 
    'create_barkopedia_gender_dataset',
    'SimpleBarkopediaDataset', 
    'SimpleDatasetConfig',
    'SimpleBarkopediaAgeDataset', 
    'create_simple_barkopedia_dataset',
    'SimpleBarkopediaGenderDataset', 
    'create_simple_barkopedia_gender_dataset'
]
