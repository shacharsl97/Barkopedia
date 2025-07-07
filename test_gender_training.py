#!/usr/bin/env python3
"""
Test script for the gender classification training.
"""

import os
import sys

def test_training_script():
    """Test if the training script can be imported and basic functionality works."""
    print("=== Testing Gender Classification Training Script ===\n")
    
    # Test import
    try:
        sys.path.append('/home/cs/weidena1/Barkopedia/TASK2')
        import train_gender_ast
        print("✓ Training script imported successfully")
    except Exception as e:
        print(f"✗ Failed to import training script: {e}")
        return False
    
    # Test argument parsing
    try:
        # Mock sys.argv to test argument parsing
        original_argv = sys.argv
        sys.argv = ['train_gender_ast.py', '--debug', '--num_epochs', '1', '--batch_size', '2']
        
        args = train_gender_ast.parse_arguments()
        print("✓ Argument parsing works")
        print(f"  - Debug mode: {args.debug}")
        print(f"  - Epochs: {args.num_epochs}")
        print(f"  - Batch size: {args.batch_size}")
        
        sys.argv = original_argv
        
    except Exception as e:
        print(f"✗ Argument parsing failed: {e}")
        sys.argv = original_argv
        return False
    
    # Test device setup
    try:
        device = train_gender_ast.setup_device('auto')
        print(f"✓ Device setup works: {device}")
    except Exception as e:
        print(f"✗ Device setup failed: {e}")
        return False
    
    print("\n🎉 Training script is ready to use!")
    print("\nTo run training:")
    print("  cd /home/cs/weidena1/Barkopedia/TASK2")
    print("  python train_gender_ast.py --debug --num_epochs 1 --batch_size 4")
    print("\nFor full training:")
    print("  python train_gender_ast.py --num_epochs 10 --batch_size 16 --hf_token YOUR_TOKEN")
    
    return True

if __name__ == "__main__":
    test_training_script()
