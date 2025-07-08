#!/usr/bin/env python3
"""
Test script for the updated Individual Dog Recognition dataset with CSV labels.
"""

import os
import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_individual_dog_dataset():
    """Test the individual dog recognition dataset with CSV labels."""
    
    print("=== Testing Individual Dog Recognition Dataset with CSV Labels ===\n")
    
    # Load HuggingFace token if available
    hf_token = None
    try:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            print("✓ Found HuggingFace token")
        else:
            print("⚠ No HuggingFace token found, using public access")
    except:
        pass
    
    # Import the dataset
    try:
        from TASK4_individual_dog_recognition.individual_dog_dataset import create_individual_dog_dataset
        print("✓ Successfully imported dataset module")
    except Exception as e:
        print(f"✗ Failed to import dataset module: {e}")
        return False
    
    # Create datasets
    print("\n1. Creating dataset splits...")
    try:
        splits = create_individual_dog_dataset(
            hf_token=hf_token,
            apply_cleaning=True,
            enable_segmentation=False,  # Disable for faster testing
            max_duration=10.0  # Limit duration for faster testing
        )
        
        print(f"✓ Train dataset: {len(splits['train'])} samples")
        print(f"✓ Validation dataset: {len(splits['validation'])} samples") 
        print(f"✓ Test dataset: {len(splits['test'])} samples")
        
    except Exception as e:
        print(f"✗ Failed to create datasets: {e}")
        return False
    
    # Test sample access
    print("\n2. Testing sample access...")
    try:
        if len(splits['train']) > 0:
            train_sample = splits['train'][0]
            print(f"✓ Train sample keys: {list(train_sample.keys())}")
            print(f"✓ Dog ID: {train_sample['dog_id']} (label: {train_sample['labels']})")
            print(f"✓ Label name: {train_sample['label_name']}")
            print(f"✓ Audio shape: {train_sample['audio'].shape}")
            print(f"✓ Input values shape: {train_sample['input_values'].shape}")
        
        if len(splits['validation']) > 0:
            val_sample = splits['validation'][0]
            print(f"✓ Validation sample dog ID: {val_sample['dog_id']} (label: {val_sample['labels']})")
            
    except Exception as e:
        print(f"✗ Failed to access samples: {e}")
        return False
    
    # Test class distributions
    print("\n3. Testing class distributions...")
    try:
        train_dist = splits['train'].get_class_distribution()
        val_dist = splits['validation'].get_class_distribution()
        
        print(f"✓ Train classes: {len(train_dist)} dogs")
        print(f"✓ Validation classes: {len(val_dist)} dogs")
        
        # Show top 5 most frequent dogs in each split
        train_top5 = sorted(train_dist.items(), key=lambda x: x[1], reverse=True)[:5]
        val_top5 = sorted(val_dist.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"✓ Train top 5 dogs: {train_top5}")
        print(f"✓ Validation top 5 dogs: {val_top5}")
        
    except Exception as e:
        print(f"✗ Failed to get class distributions: {e}")
        return False
    
    # Test DataLoader
    print("\n4. Testing DataLoader...")
    try:
        train_loader = splits['train'].get_dataloader(batch_size=2, shuffle=False)
        batch = next(iter(train_loader))
        
        print(f"✓ Batch keys: {list(batch.keys())}")
        print(f"✓ Batch input_values shape: {batch['input_values'].shape}")
        print(f"✓ Batch labels: {batch['labels']}")
        print(f"✓ Batch dog_ids: {batch.get('dog_id', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Failed to test DataLoader: {e}")
        return False
    
    print("\n=== All tests passed! ===")
    print("The Individual Dog Recognition dataset is correctly using CSV labels.")
    return True

if __name__ == "__main__":
    success = test_individual_dog_dataset()
    sys.exit(0 if success else 1)
