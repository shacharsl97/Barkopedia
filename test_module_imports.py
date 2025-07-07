#!/usr/bin/env python3
"""
Test script to verify all imports work correctly after module rename.
"""

def test_imports():
    print("=== Testing Barkopedia Datasets Module ===\n")
    
    # Test simple datasets
    print("1. Testing simple dataset imports...")
    try:
        from barkopedia_datasets import SimpleBarkopediaGenderDataset, SimpleBarkopediaAgeDataset
        print("   ✓ Simple datasets imported successfully")
    except Exception as e:
        print(f"   ✗ Error importing simple datasets: {e}")
        return False
    
    # Test full datasets
    print("2. Testing full dataset imports...")
    try:
        from barkopedia_datasets import BarkopediaGenderDataset, BarkopediaAgeGroupDataset
        print("   ✓ Full datasets imported successfully")
    except Exception as e:
        print(f"   ✗ Error importing full datasets: {e}")
        return False
    
    # Test create functions
    print("3. Testing create functions...")
    try:
        from barkopedia_datasets import (
            create_simple_barkopedia_dataset, 
            create_simple_barkopedia_gender_dataset,
            create_barkopedia_age_dataset,
            create_barkopedia_gender_dataset
        )
        print("   ✓ Create functions imported successfully")
    except Exception as e:
        print(f"   ✗ Error importing create functions: {e}")
        return False
    
    # Test interfaces
    print("4. Testing interfaces...")
    try:
        from barkopedia_datasets import (
            SimpleBarkopediaDataset, 
            SimpleDatasetConfig,
            BarkopediaDataset, 
            DatasetConfig
        )
        print("   ✓ Interfaces imported successfully")
    except Exception as e:
        print(f"   ✗ Error importing interfaces: {e}")
        return False
    
    print("\n=== All imports successful! ===")
    return True

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 Module rename completed successfully!")
        print("You can now use:")
        print("  from barkopedia_datasets import SimpleBarkopediaGenderDataset")
        print("  from barkopedia_datasets import SimpleBarkopediaAgeDataset")
        print("  from barkopedia_datasets import BarkopediaGenderDataset")
        print("  from barkopedia_datasets import BarkopediaAgeGroupDataset")
    else:
        print("\n❌ Some imports failed. Please check the errors above.")
