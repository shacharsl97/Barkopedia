#!/usr/bin/env python3
"""
Test script for Barkopedia gender classification datasets.
Tests both the simple and full dataset implementations.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_simple_gender_dataset():
    """Test the simple gender dataset implementation."""
    print("=== Testing Simple Barkopedia Gender Dataset ===\n")
    
    try:
        from barkopedia_datasets.simple_barkopedia_gender_dataset import create_simple_barkopedia_gender_dataset
        
        # Load HuggingFace token if available
        hf_token = os.environ.get("HF_TOKEN")
        
        # Create dataset with cleaning enabled
        print("1. Creating simple dataset splits...")
        splits = create_simple_barkopedia_gender_dataset(
            hf_token=hf_token,
            apply_cleaning=True,
            max_duration=5.0  # Limit to 5 seconds max
        )
        
        train_dataset = splits["train"]
        test_dataset = splits["test"]
        
        print(f"   ✓ Train dataset: {len(train_dataset)} samples")
        print(f"   ✓ Test dataset: {len(test_dataset)} samples")
        
        # Test sample access
        print("\n2. Testing sample access...")
        train_sample = train_dataset[0]
        
        print(f"   ✓ Sample keys: {list(train_sample.keys())}")
        print(f"   ✓ Input values shape: {train_sample['input_values'].shape}")
        print(f"   ✓ Label: {train_sample['labels']} ({train_sample['label_name']})")
        print(f"   ✓ Sampling rate: {train_sample['sampling_rate']}")
        
        # Test class distribution
        print("\n3. Class distribution:")
        train_dist = train_dataset.get_class_distribution()
        for label, count in train_dist.items():
            print(f"   {label}: {count} samples")
        
        # Test DataLoader
        print("\n4. Testing DataLoader...")
        dataloader = train_dataset.get_dataloader(batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        
        print(f"   ✓ Batch input_values shape: {batch['input_values'].shape}")
        print(f"   ✓ Batch labels shape: {batch['labels'].shape}")
        
        print("\n=== Simple Gender Dataset Tests Passed! ===\n")
        return True
        
    except Exception as e:
        print(f"❌ Simple gender dataset test failed: {e}")
        return False


def test_full_gender_dataset():
    """Test the full gender dataset implementation."""
    print("=== Testing Full Barkopedia Gender Dataset ===\n")
    
    try:
        from barkopedia_datasets.barkopedia_gender_dataset import create_barkopedia_gender_dataset
        
        # Load HuggingFace token if available
        hf_token = os.environ.get("HF_TOKEN")
        
        # Create dataset
        print("1. Creating full dataset splits...")
        splits = create_barkopedia_gender_dataset(
            hf_token=hf_token,
            augmentation=False  # Start without augmentation for testing
        )
        
        train_dataset = splits["train"]
        test_dataset = splits["test"]
        
        print(f"   ✓ Train dataset: {len(train_dataset)} samples")
        print(f"   ✓ Test dataset: {len(test_dataset)} samples")
        
        # Test sample processing
        print("\n2. Testing sample processing...")
        train_sample = train_dataset[0]
        
        print(f"   ✓ Sample keys: {list(train_sample.keys())}")
        print(f"   ✓ Input values shape: {train_sample['input_values'].shape}")
        print(f"   ✓ Label: {train_sample['labels']} ({train_sample['label_name']})")
        print(f"   ✓ Sampling rate: {train_sample['sampling_rate']}")
        
        # Test class distribution
        print("\n3. Class distribution:")
        train_dist = train_dataset.get_class_distribution()
        for label, count in train_dist.items():
            print(f"   {label}: {count} samples")
        
        # Test DataLoader
        print("\n4. Testing DataLoader...")
        dataloader = train_dataset.get_dataloader(batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        
        print(f"   ✓ Batch input_values shape: {batch['input_values'].shape}")
        print(f"   ✓ Batch labels shape: {batch['labels'].shape}")
        
        print("\n=== Full Gender Dataset Tests Passed! ===\n")
        return True
        
    except Exception as e:
        print(f"❌ Full gender dataset test failed: {e}")
        return False


def test_import_functionality():
    """Test that all datasets can be imported from the package."""
    print("=== Testing Import Functionality ===\n")
    
    try:
        # Test simple imports
        from datasets import (
            SimpleBarkopediaGenderDataset, 
            create_simple_barkopedia_gender_dataset,
            BarkopediaGenderDataset,
            create_barkopedia_gender_dataset
        )
        
        print("   ✓ All gender dataset classes imported successfully")
        
        # Test age dataset imports still work
        from datasets import (
            SimpleBarkopediaAgeDataset,
            create_simple_barkopedia_dataset,
            BarkopediaAgeGroupDataset,
            create_barkopedia_age_dataset
        )
        
        print("   ✓ All age dataset classes imported successfully")
        print("\n=== Import Tests Passed! ===\n")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Barkopedia Gender Dataset Tests\n")
    
    # Run all tests
    results = []
    
    results.append(test_import_functionality())
    results.append(test_simple_gender_dataset())
    results.append(test_full_gender_dataset())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
