#!/usr/bin/env python3
"""
Quick Dataset Creator - Automatic mode only
"""

from dataset_creator import DatasetCreator

def main():
    creator = DatasetCreator()
    
    input_folder = "/Users/talhaansari/Developer/Adobe/1a new/input"
    dataset_file = "/Users/talhaansari/Developer/Adobe/1a new/training_dataset.csv"
    
    print("Creating dataset with automatic labeling...")
    
    # Create dataset with automatic labeling only
    dataset_df = creator.create_comprehensive_dataset(
        input_folder, 
        dataset_file, 
        interactive=False  # Use automatic labeling only
    )
    
    if dataset_df is not None:
        print("\n✅ Dataset created successfully!")
        
        # Add synthetic examples
        print("Adding synthetic examples...")
        creator.augment_dataset_with_synthetic_examples(dataset_file)
        
        print(f"\n📊 Dataset ready for training at: {dataset_file}")
    else:
        print("❌ Failed to create dataset")

if __name__ == "__main__":
    main()
