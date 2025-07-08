#!/usr/bin/env python3
"""
Quick test script for multi-model training.
"""

import os
import sys
import subprocess
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_creation():
    """Test that all models can be created."""
    print("=== Testing Model Creation ===")
    
    try:
        from models.ast_classification_model import ASTClassificationModel
        from models.wav2vec2_classification_model import create_wav2vec2_model
        
        print("✓ AST model import successful")
        print("✓ Wav2Vec2 model import successful")
        
        # Test model creation
        ast_model = ASTClassificationModel(num_labels=2)
        print("✓ AST model created")
        
        wav2vec2_custom = create_wav2vec2_model(num_labels=2, model_type='custom')
        print("✓ Wav2Vec2 custom model created")
        
        wav2vec2_pretrained = create_wav2vec2_model(num_labels=2, model_type='pretrained')
        print("✓ Wav2Vec2 pretrained model created")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_training_script():
    """Test the training script with minimal parameters."""
    print("\n=== Testing Training Script ===")
    
    # Test with very small parameters
    cmd = [
        "python", "train_multi_model.py",
        "--model_type", "ast",
        "--num_epochs", "1",
        "--batch_size", "2",
        "--eval_steps", "50",
        "--save_steps", "100",
        "--debug",
        "--device", "cuda:0"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Training script test passed")
            return True
        else:
            print(f"✗ Training script test failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Training script test timed out")
        return False
    except Exception as e:
        print(f"✗ Training script test error: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting multi-model training tests...\n")
    
    # Test model creation
    if not test_model_creation():
        print("Model creation test failed. Exiting.")
        return False
    
    # Test training script
    if not test_training_script():
        print("Training script test failed. Exiting.")
        return False
    
    print("\n✓ All tests passed! Ready for full training.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
