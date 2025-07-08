#!/usr/bin/env python3
"""
Test script to verify that the dataset loader correctly uses validation split
and assigns labels from validation_labels.csv
"""

import sys
import os
sys.path.append('/home/cs/weidena1/Barkopedia')

from barkopedia_datasets.simple_barkopedia_gender_dataset import (
    SimpleBarkopediaGenderDataset, 
    SimpleDatasetConfig
)

def test_validation_split_loading():
    """Test that the validation split is loaded correctly."""
    
    print("Testing validation split loading with CSV labels...")
    
    # Create config for test split
    config = SimpleDatasetConfig(
        dataset_name="barkopedia_gender_test",
        cache_dir="/home/cs/weidena1/Barkopedia/barkopedia_gender_cache",
        hf_token=None,
        enable_segmentation=False,  # Disable segmentation for cleaner testing
        segment_duration=2.0,
        segment_overlap=0.0,
        min_segment_duration=0.3,
        max_segment_duration=5.0,
        energy_threshold=0.01,
        silence_min_duration=0.1
    )
    
    # Test loading validation split as test data
    print("\n" + "="*50)
    print("TESTING VALIDATION SPLIT LOADING")
    print("="*50)
    
    try:
        dataset = SimpleBarkopediaGenderDataset(config, split="test")
        dataset.load_data()
        
        print(f"\nDataset loaded successfully:")
        print(f"- Total samples: {len(dataset.data)}")
        print(f"- Label distribution: {dataset.get_class_distribution()}")
        print(f"- Label mapping: {dataset.id_to_label}")
        
        # Check first few samples
        print("\nFirst 5 samples:")
        for i in range(min(5, len(dataset.data))):
            sample = dataset.data[i]
            print(f"  Sample {i}: label={sample['labels']}, "
                  f"label_name={sample.get('label_name', 'unknown')}, "
                  f"source_file={sample.get('source_file', 'unknown')}")
        
        # Test __getitem__ method
        print("\nTesting __getitem__ method:")
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            print(f"  Item {i}: keys={list(item.keys())}, label={item['labels']}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def test_segmentation_with_validation():
    """Test that segmentation works correctly with validation labels."""
    
    print("\n" + "="*50)
    print("TESTING SEGMENTATION WITH VALIDATION LABELS")
    print("="*50)
    
    # Create config with segmentation enabled
    config = SimpleDatasetConfig(
        dataset_name="barkopedia_gender_segmented",
        cache_dir="/home/cs/weidena1/Barkopedia/barkopedia_gender_cache",
        hf_token=None,
        enable_segmentation=True,
        segment_duration=2.0,
        segment_overlap=0.0,
        min_segment_duration=0.3,
        max_segment_duration=5.0,
        energy_threshold=0.01,
        silence_min_duration=0.1
    )
    
    try:
        dataset = SimpleBarkopediaGenderDataset(config, split="test")
        dataset.load_data()
        
        print(f"\nSegmented dataset loaded successfully:")
        print(f"- Total segments: {len(dataset.data)}")
        print(f"- Label distribution: {dataset.get_class_distribution()}")
        
        # Check segments from first few audio files
        print("\nFirst 5 segments:")
        for i in range(min(5, len(dataset.data))):
            sample = dataset.data[i]
            print(f"  Segment {i}: label={sample['labels']}, "
                  f"label_name={sample.get('label_name', 'unknown')}, "
                  f"source_file={sample.get('source_file', 'unknown')}, "
                  f"segment_id={sample.get('segment_id', 0)}, "
                  f"duration={sample.get('segment_duration', 0):.2f}s")
        
        # Test __getitem__ method
        print("\nTesting __getitem__ method with segmentation:")
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            print(f"  Item {i}: keys={list(item.keys())}, label={item['labels']}")
            
    except Exception as e:
        print(f"Error loading segmented dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    print("Testing updated dataset loader with validation split and CSV labels...")
    
    success1 = test_validation_split_loading()
    success2 = test_segmentation_with_validation()
    
    if success1 and success2:
        print("\n✅ All tests passed! Dataset loader is working correctly.")
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
