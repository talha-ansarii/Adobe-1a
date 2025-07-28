#!/usr/bin/env python3
"""
Debug script to test the PDF extractor step by step
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import fitz

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

from pdf_to_json_extractor import PDFToJSONExtractor

def debug_extraction():
    print("🔧 Debug: PDF to JSON Extractor")
    print("=" * 50)
    
    # Initialize extractor
    extractor = PDFToJSONExtractor()
    
    # Test with example PDF
    pdf_path = "input/example.pdf"
    
    # Extract text elements
    print("\n1. Extracting text elements...")
    df = extractor.extract_text_elements(pdf_path)
    print(f"   Extracted {len(df)} elements")
    print(f"   Columns: {list(df.columns)}")
    
    # Engineer features
    print("\n2. Engineering features...")
    df_features = extractor.engineer_features(df.copy())
    print(f"   Total features: {len(df_features.columns)}")
    print(f"   Feature columns: {list(df_features.columns)}")
    
    # Check model
    print("\n3. Model information...")
    model_data = joblib.load(extractor.model_path)
    print(f"   Available keys: {list(model_data.keys())}")
    
    feature_selector = model_data['feature_selector']
    print(f"   Expected features: {feature_selector.feature_names_in_}")
    print(f"   Selected features count: {feature_selector.n_features_}")
    
    # Check which features are missing
    expected = feature_selector.feature_names_in_
    available = df_features.columns
    missing = [f for f in expected if f not in available]
    extra = [f for f in available if f not in expected]
    
    print(f"   Missing features: {missing}")
    print(f"   Extra features (first 10): {list(extra)[:10]}")
    
    # Try prediction on a small sample
    print("\n4. Testing prediction...")
    sample_df = df_features.head(5).copy()
    
    # Add missing features with defaults
    for feature in missing:
        sample_df[feature] = 0
    
    # Select expected features
    X = sample_df[expected]
    print(f"   Input shape: {X.shape}")
    
    # Apply feature selection
    X_selected = feature_selector.transform(X)
    print(f"   Selected shape: {X_selected.shape}")
    
    # Make prediction
    model = model_data['ensemble_classifier']
    predictions = model.predict(X_selected)
    probabilities = model.predict_proba(X_selected)
    
    print(f"   Predictions: {predictions}")
    print(f"   Probabilities: {probabilities[:, 1]}")
    
    # Check some text samples
    print(f"   Sample texts:")
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        print(f"     {i}: '{row['text'][:50]}...' -> {predictions[i]} ({probabilities[i, 1]:.3f})")

if __name__ == "__main__":
    debug_extraction()
