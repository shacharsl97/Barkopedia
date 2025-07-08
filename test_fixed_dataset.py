#!/usr/bin/env python3
"""
Quick test of the fixed dataset loading.
"""

import sys
import os
sys.path.append('/home/cs/weidena1/Barkopedia')

from barkopedia_datasets import create_simple_barkopedia_gender_dataset

def test_dataset_loading():
    print("=== Testing Fixed Dataset Loading ===\n")
    
    # Load HuggingFace token if available
    hf_token = None
    try:
        hf_token = os.environ.get("HF_TOKEN")
    except:
        pass
    
    try:
        # Test with minimal settings first (no segmentation, no cleaning)
        print("1. Testing basic dataset loading (no segmentation, no cleaning)...")
        splits = create_simple_barkopedia_gender_dataset(
            hf_token=hf_token,
            apply_cleaning=False,
            enable_segmentation=False,
            cache_dir="./test_barkopedia_cache"
        )
        
        train_dataset = splits["train"]
        validation_dataset = splits["validation"]
        test_dataset = splits["test"]
        
        print(f"   ✓ Train dataset: {len(train_dataset)} samples")
        print(f"   ✓ Validation dataset: {len(validation_dataset)} samples")
        print(f"   ✓ Test dataset: {len(test_dataset)} samples")
        
        # Check class distributions
        print("\n2. Checking class distributions...")
        train_dist = train_dataset.get_class_distribution()
        validation_dist = validation_dataset.get_class_distribution()
        
        print(f"   Train distribution: {train_dist}")
        print(f"   Validation distribution: {validation_dist}")
        
        # Test a sample
        print("\n3. Testing sample access...")
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"   ✓ Sample keys: {list(sample.keys())}")
            print(f"   ✓ Label: {sample['labels']} ({sample['label_name']})")
            print(f"   ✓ Audio shape: {sample['audio'].shape}")
            print(f"   ✓ Input values shape: {sample['input_values'].shape}")
        
        if len(validation_dataset) > 0:
            val_sample = validation_dataset[0]
            print(f"   ✓ Validation sample label: {val_sample['labels']} ({val_sample['label_name']})")
        
        print("\n✅ Dataset loading test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dataset_loading()
