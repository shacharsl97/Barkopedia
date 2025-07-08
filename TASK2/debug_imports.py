#!/usr/bin/env python3
"""
Debug script to test imports and find the config module issue.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== Debugging Config Module Issue ===\n")

print("1. Testing basic imports...")
try:
    import torch
    print("✓ torch imported successfully")
except Exception as e:
    print(f"✗ torch import failed: {e}")

try:
    import transformers
    print("✓ transformers imported successfully")
except Exception as e:
    print(f"✗ transformers import failed: {e}")

try:
    import datasets
    print("✓ datasets imported successfully")
except Exception as e:
    print(f"✗ datasets import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing our custom modules...")
try:
    from barkopedia_datasets import SimpleBarkopediaGenderDataset
    print("✓ SimpleBarkopediaGenderDataset imported successfully")
except Exception as e:
    print(f"✗ SimpleBarkopediaGenderDataset import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from models.ast_classification_model import ASTClassificationModel
    print("✓ ASTClassificationModel imported successfully")
except Exception as e:
    print(f"✗ ASTClassificationModel import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing dataset creation...")
try:
    from barkopedia_datasets import create_simple_barkopedia_gender_dataset
    print("✓ create_simple_barkopedia_gender_dataset imported successfully")
except Exception as e:
    print(f"✗ create_simple_barkopedia_gender_dataset import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debug complete ===")
