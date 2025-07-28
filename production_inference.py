#!/usr/bin/env python3
"""
Production PDF Heading Detection
Uses trained model to detect headings in new PDFs
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from train_real_model import ProductionHeadingDetector
import argparse


class PDFHeadingPredictor:
    """Production-ready PDF heading predictor"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.detector = ProductionHeadingDetector(model_path)
        
        # Load the trained model
        if not self.detector.load_model():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    def analyze_pdf(self, pdf_path, output_file=None, confidence_threshold=0.5):
        """Analyze a single PDF and detect headings"""
        print(f"Analyzing: {pdf_path}")
        
        try:
            # Predict headings
            result_df = self.detector.predict_pdf_headings(pdf_path, use_ensemble=True)
            
            if result_df.empty:
                print("❌ No text found in PDF")
                return None
            
            # Filter by confidence threshold
            high_confidence_headings = result_df[
                (result_df['predicted_heading'] == 1) & 
                (result_df['heading_probability'] >= confidence_threshold)
            ]
            
            print(f"📄 Total text elements: {len(result_df)}")
            print(f"🎯 Predicted headings: {len(high_confidence_headings)}")
            print(f"📊 Average confidence: {high_confidence_headings['heading_probability'].mean():.1%}")
            
            # Display headings
            if len(high_confidence_headings) > 0:
                print(f"\n🔍 Detected Headings (confidence ≥ {confidence_threshold:.0%}):")
                print("-" * 60)
                
                for i, (_, heading) in enumerate(high_confidence_headings.iterrows(), 1):
                    confidence = heading['heading_probability']
                    text = heading['text']
                    page = heading.get('page_number', '?')
                    font_size = heading.get('font_size', 0)
                    is_bold = heading.get('is_bold', False)
                    
                    print(f"{i:2d}. {text}")
                    print(f"     📍 Page {page} | 📊 {confidence:.1%} | 🔤 {font_size:.1f}pt | {'Bold' if is_bold else 'Normal'}")
                    print()
            else:
                print("❌ No headings detected with sufficient confidence")
            
            # Save results if requested
            if output_file:
                result_df.to_csv(output_file, index=False)
                print(f"💾 Results saved to: {output_file}")
            
            return result_df
            
        except Exception as e:
            print(f"❌ Error analyzing PDF: {str(e)}")
            return None
    
    def analyze_folder(self, folder_path, output_folder=None, confidence_threshold=0.5):
        """Analyze all PDFs in a folder"""
        folder_path = Path(folder_path)
        pdf_files = list(folder_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {folder_path}")
            return
        
        print(f"📁 Found {len(pdf_files)} PDF files")
        print("=" * 80)
        
        results_summary = []
        
        for pdf_file in pdf_files:
            print(f"\n{'='*60}")
            print(f"Processing: {pdf_file.name}")
            print('='*60)
            
            # Set output file if output folder specified
            output_file = None
            if output_folder:
                output_folder = Path(output_folder)
                output_folder.mkdir(exist_ok=True)
                output_file = output_folder / f"{pdf_file.stem}_headings.csv"
            
            # Analyze PDF
            result_df = self.analyze_pdf(str(pdf_file), output_file, confidence_threshold)
            
            if result_df is not None:
                # Collect summary statistics
                total_elements = len(result_df)
                predicted_headings = len(result_df[result_df['predicted_heading'] == 1])
                high_conf_headings = len(result_df[
                    (result_df['predicted_heading'] == 1) & 
                    (result_df['heading_probability'] >= confidence_threshold)
                ])
                
                results_summary.append({
                    'pdf_file': pdf_file.name,
                    'total_elements': total_elements,
                    'predicted_headings': predicted_headings,
                    'high_confidence_headings': high_conf_headings,
                    'heading_ratio': high_conf_headings / total_elements if total_elements > 0 else 0
                })
        
        # Print summary
        print(f"\n{'='*80}")
        print("BATCH PROCESSING SUMMARY")
        print('='*80)
        
        summary_df = pd.DataFrame(results_summary)
        if not summary_df.empty:
            print(summary_df.to_string(index=False))
            
            print(f"\n📊 Overall Statistics:")
            print(f"Total PDFs processed: {len(summary_df)}")
            print(f"Average headings per document: {summary_df['high_confidence_headings'].mean():.1f}")
            print(f"Total headings detected: {summary_df['high_confidence_headings'].sum()}")
            
            # Save summary
            if output_folder:
                summary_file = Path(output_folder) / "batch_processing_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                print(f"💾 Summary saved to: {summary_file}")
    
    def get_model_info(self):
        """Display information about the loaded model"""
        print("🤖 Model Information:")
        print("-" * 30)
        print(f"Model file: {self.model_path}")
        print(f"Selected features: {len(self.detector.selected_features)}")
        print(f"Feature names: {', '.join(self.detector.selected_features[:5])}{'...' if len(self.detector.selected_features) > 5 else ''}")
        
        # Feature importance if available
        if hasattr(self.detector.dt_classifier, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.detector.selected_features,
                'importance': self.detector.dt_classifier.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🎯 Top 5 Most Important Features:")
            for _, row in importance_df.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="PDF Heading Detection using Trained Model")
    parser.add_argument("input", help="Input PDF file or folder")
    parser.add_argument("-o", "--output", help="Output folder for results")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, 
                       help="Confidence threshold (0.0-1.0, default: 0.5)")
    parser.add_argument("-m", "--model", default="/Users/talhaansari/Developer/Adobe/1a new/production_heading_model.pkl",
                       help="Path to trained model file")
    parser.add_argument("--info", action="store_true", help="Show model information")
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        print("Please run train_real_model.py first to train the model.")
        sys.exit(1)
    
    try:
        # Initialize predictor
        predictor = PDFHeadingPredictor(args.model)
        
        # Show model info if requested
        if args.info:
            predictor.get_model_info()
            return
        
        # Check input exists
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ Input not found: {args.input}")
            sys.exit(1)
        
        # Process input
        if input_path.is_file() and input_path.suffix.lower() == '.pdf':
            # Single PDF file
            output_file = None
            if args.output:
                output_folder = Path(args.output)
                output_folder.mkdir(exist_ok=True)
                output_file = output_folder / f"{input_path.stem}_headings.csv"
            
            predictor.analyze_pdf(str(input_path), output_file, args.confidence)
            
        elif input_path.is_dir():
            # Folder of PDFs
            predictor.analyze_folder(str(input_path), args.output, args.confidence)
            
        else:
            print(f"❌ Input must be a PDF file or folder: {args.input}")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode if no arguments
        print("PDF Heading Detection - Production System")
        print("=" * 50)
        
        model_path = "/Users/talhaansari/Developer/Adobe/1a new/production_heading_model.pkl"
        
        if not Path(model_path).exists():
            print(f"❌ Trained model not found: {model_path}")
            print("Please run the following steps:")
            print("1. python dataset_creator.py  # Create training dataset")
            print("2. python train_real_model.py  # Train the model")
            print("3. python production_inference.py  # Use this script")
            sys.exit(1)
        
        try:
            predictor = PDFHeadingPredictor(model_path)
            predictor.get_model_info()
            
            print(f"\n📁 Available PDFs in input folder:")
            input_folder = Path("/Users/talhaansari/Developer/Adobe/1a new/input")
            pdf_files = list(input_folder.glob("*.pdf"))
            
            if pdf_files:
                for i, pdf_file in enumerate(pdf_files, 1):
                    print(f"{i}. {pdf_file.name}")
                
                # Ask user what to do
                print(f"\nOptions:")
                print("1. Analyze all PDFs in input folder")
                print("2. Analyze specific PDF")
                print("3. Exit")
                
                choice = input("\nChoose option (1-3): ").strip()
                
                if choice == '1':
                    predictor.analyze_folder(str(input_folder), confidence_threshold=0.5)
                elif choice == '2':
                    pdf_num = int(input(f"Enter PDF number (1-{len(pdf_files)}): ")) - 1
                    if 0 <= pdf_num < len(pdf_files):
                        predictor.analyze_pdf(str(pdf_files[pdf_num]))
                    else:
                        print("Invalid PDF number")
                elif choice == '3':
                    print("Goodbye!")
                else:
                    print("Invalid choice")
            else:
                print("No PDF files found in input folder")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    else:
        main()
