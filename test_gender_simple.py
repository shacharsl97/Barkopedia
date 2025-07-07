#!/usr/bin/env python3
"""
Simple test script for the gender classification dataset.
"""

import os

def test_gender_dataset():
    print("=== Testing Barkopedia Gender Classification Dataset ===\n")
    
    # Test import
    try:
        from barkopedia_datasets import SimpleBarkopediaGenderDataset, create_simple_barkopedia_gender_dataset
        print("✓ Gender dataset imported successfully")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return
    
    # Test creating a simple config (without actually loading data)
    try:
        from barkopedia_datasets import SimpleDatasetConfig
        
        config = SimpleDatasetConfig(
            dataset_name="test_gender",
            cache_dir="./test_cache",
            apply_cleaning=True,
            sampling_rate=16000
        )
        
        # Create dataset instance (but don't load data yet)
        dataset = SimpleBarkopediaGenderDataset(config, split="train")
        
        print("✓ Dataset instance created successfully")
        print(f"  - Dataset name: {config.dataset_name}")
        print(f"  - Cleaning enabled: {config.apply_cleaning}")
        print(f"  - Sampling rate: {config.sampling_rate}")
        print(f"  - Split: {dataset.split}")
        print(f"  - Label mapping: {dataset.id_to_label}")
        
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return
    
    print("\n🎉 Gender dataset is ready to use!")
    print("\nTo actually load and use the dataset:")
    print("  splits = create_simple_barkopedia_gender_dataset(hf_token=your_token)")
    print("  train_dataset = splits['train']")
    print("  test_dataset = splits['test']")

if __name__ == "__main__":
    test_gender_dataset()
