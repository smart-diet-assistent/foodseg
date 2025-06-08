#!/usr/bin/env python3

"""
Dataset preparation script for FoodSeg103.
Downloads and prepares the dataset for training.
"""

import os
import sys
from dataset import prepare_dataset

def main():
    """Main function to prepare the FoodSeg103 dataset."""
    print("="*50)
    print("FoodSeg103 Dataset Preparation")
    print("="*50)
    
    try:
        # Prepare dataset
        dataset, label_mapping = prepare_dataset()
        
        print("\nDataset preparation completed successfully!")
        print(f"Dataset info:")
        print(f"- Train samples: {len(dataset['train'])}")
        
        if 'validation' in dataset:
            print(f"- Validation samples: {len(dataset['validation'])}")
        if 'test' in dataset:
            print(f"- Test samples: {len(dataset['test'])}")
        
        # Show label mapping info
        if label_mapping:
            print(f"\nLabel mapping applied:")
            print(f"- Original labels: {list(label_mapping.keys())}")
            print(f"- Mapped to: {list(label_mapping.values())}")
            print(f"- Number of classes after mapping: {len(label_mapping)}")
        else:
            print(f"\nNo label filtering applied, using all original labels")
        
        # Show sample data structure
        sample = dataset['train'][0]
        print(f"\nSample data structure:")
        for key in sample.keys():
            if hasattr(sample[key], 'size'):
                print(f"- {key}: {type(sample[key])} with size {sample[key].size}")
            else:
                print(f"- {key}: {type(sample[key])}")
        
        print(f"\nDataset is ready for training!")
        
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
