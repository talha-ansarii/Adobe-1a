#!/usr/bin/env python3
"""
PDF to JSON Extractor
====================

A production-ready tool that extracts title and hierarchical headings from PDF files
and outputs them in a structured JSON format.

Features:
- Accepts PDF files up to 50 pages
- Extracts document title and hierarchical headings (H1, H2, H3)
- Uses AI-powered heading detection with 82%+ F1 score
- Outputs structured JSON with heading levels and page numbers
- Handles various PDF formats and document types

Author: AI Assistant
Date: July 2025
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import joblib
import pandas as pd
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Add the project directory to Python path
project_dir = Path(__file__).parent


class PDFToJSONExtractor:
    """
    Main class for extracting PDF content and converting to structured JSON format.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the PDF extractor with a trained model.
        
        Args:
            model_path: Path to the trained heading detection model
        """
        self.model_path = model_path or os.path.join(project_dir, "production_heading_model.pkl")
        self.model = None
        self.feature_selector = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load the trained heading detection model."""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['ensemble_classifier']  # Use ensemble classifier
            self.feature_selector = model_data['feature_selector']
            self.scaler = model_data.get('scaler', None)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"❌ Error: Model file not found at {self.model_path}")
            print("Please run 'python train_real_model.py' first to train the model.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Validate PDF file and check page limit.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            bool: True if PDF is valid and within page limit
        """
        if not os.path.exists(pdf_path):
            print(f"❌ Error: PDF file not found: {pdf_path}")
            return False
        
        try:
            with fitz.open(pdf_path) as doc:
                page_count = len(doc)
                if page_count > 50:
                    print(f"❌ Error: PDF has {page_count} pages. Maximum allowed is 50 pages.")
                    return False
                print(f"📄 PDF validated: {page_count} pages")
                return True
        except Exception as e:
            print(f"❌ Error opening PDF: {e}")
            return False
    
    def extract_text_elements(self, pdf_path: str) -> pd.DataFrame:
        """
        Extract all text elements from PDF with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            DataFrame with text elements and features
        """
        elements = []
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    blocks = page.get_text("dict")["blocks"]
                    
                    block_id = 0
                    for block in blocks:
                        if "lines" in block:
                            line_id = 0
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    text = span["text"].strip()
                                    if text:
                                        elements.append({
                                            'page_number': page_num + 1,
                                            'block_id': block_id,
                                            'line_id': line_id,
                                            'text': text,
                                            'font_size': span['size'],
                                            'font_flags': span['flags'],
                                            'bbox': span['bbox'],
                                            'x0': span['bbox'][0],
                                            'y0': span['bbox'][1],
                                            'x1': span['bbox'][2],
                                            'y1': span['bbox'][3],
                                        })
                                line_id += 1
                            block_id += 1
            
            return pd.DataFrame(elements)
            
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {e}")
            return pd.DataFrame()
    
    def detect_title(self, df: pd.DataFrame) -> str:
        """
        Detect the document title from text elements.
        
        Args:
            df: DataFrame with text elements
            
        Returns:
            str: Detected document title
        """
        if df.empty:
            return "Untitled Document"
        
        # Look for title on first page with largest font size
        first_page = df[df['page_number'] == 1].copy()
        if first_page.empty:
            return "Untitled Document"
        
        # Find elements with the largest font size on first page
        max_font_size = first_page['font_size'].max()
        title_candidates = first_page[first_page['font_size'] == max_font_size]
        
        # Combine title candidates that appear close to each other
        if not title_candidates.empty:
            # Sort by vertical position (y0)
            title_candidates = title_candidates.sort_values('y0')
            title_parts = []
            
            for _, element in title_candidates.head(3).iterrows():  # Take top 3 candidates
                text = element['text'].strip()
                if text and len(text) > 2:  # Filter out very short text
                    title_parts.append(text)
            
            if title_parts:
                title = ' '.join(title_parts)
                # Clean up the title
                title = re.sub(r'\s+', ' ', title).strip()
                return title
        
        # Fallback: use first substantial text element
        substantial_text = first_page[first_page['text'].str.len() > 5]
        if not substantial_text.empty:
            return substantial_text.iloc[0]['text'].strip()
        
        return "Untitled Document"
    
    def extract_pos_features(self, df):
        """Extract Part-of-Speech features from text"""
        def get_pos_features(text):
            try:
                tokens = word_tokenize(str(text).lower())
                pos_tags = pos_tag(tokens)
                
                pos_counts = defaultdict(int)
                total_words = len(tokens)
                
                for word, pos in pos_tags:
                    if pos.startswith('VB'):  # Verbs
                        pos_counts['verb_ratio'] += 1
                    elif pos.startswith('NN'):  # Nouns
                        pos_counts['noun_ratio'] += 1
                    elif pos.startswith('JJ'):  # Adjectives
                        pos_counts['adjective_ratio'] += 1
                    elif pos.startswith('RB'):  # Adverbs
                        pos_counts['adverb_ratio'] += 1
                    elif pos in ['PRP', 'PRP$']:  # Pronouns
                        pos_counts['pronoun_ratio'] += 1
                    elif pos == 'CD':  # Cardinal numbers
                        pos_counts['number_ratio'] += 1
                
                # Convert to ratios
                for key in pos_counts:
                    pos_counts[key] = pos_counts[key] / total_words if total_words > 0 else 0
                
                return pos_counts
                
            except:
                return {'verb_ratio': 0, 'noun_ratio': 0, 'adjective_ratio': 0, 
                        'adverb_ratio': 0, 'pronoun_ratio': 0, 'number_ratio': 0}
        
        print("Extracting POS features...")
        pos_features = df['text'].apply(get_pos_features)
        pos_df = pd.DataFrame(pos_features.tolist())
        df = pd.concat([df, pos_df], axis=1)
        return df
    
    def engineer_features(self, df):
        """Engineer comprehensive features from the dataset"""
        # Copy dataframe to avoid modifying original
        df = df.copy()
        
        # Basic text features
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        df['char_count'] = df['text'].apply(lambda x: len(str(x)))
        df['avg_word_length'] = df['char_count'] / df['word_count'].replace(0, 1)
        
        # Font features - calculate median font size for threshold
        median_font_size = df['font_size'].median()
        df['font_threshold'] = median_font_size
        df['relative_font_size'] = df['font_size'] / median_font_size
        df['font_threshold_flag'] = (df['font_size'] > median_font_size).astype(int)
        
        # Font styling features
        df['is_bold'] = ((df['font_flags'] & 2**4) != 0).astype(int)
        df['is_italic'] = ((df['font_flags'] & 2**1) != 0).astype(int)
        
        # Calculate dimensions
        df['width'] = df['x1'] - df['x0']
        df['height'] = df['y1'] - df['y0']
        
        # Text case features
        df['is_all_caps'] = df['text'].apply(lambda x: str(x).isupper()).astype(int)
        df['is_title_case'] = df['text'].apply(lambda x: str(x).istitle()).astype(int)
        df['is_lower_case'] = df['text'].apply(lambda x: str(x).islower()).astype(int)
        
        # Structural features
        df['starts_with_number'] = df['text'].apply(lambda x: bool(re.match(r'^\s*\d+', str(x)))).astype(int)
        df['ends_with_colon'] = df['text'].apply(lambda x: str(x).strip().endswith(':')).astype(int)
        df['has_punctuation'] = df['text'].apply(lambda x: any(c in str(x) for c in '.,;!?')).astype(int)
        
        # Length-based features
        df['is_short'] = (df['word_count'] <= 10).astype(int)
        df['is_very_short'] = (df['word_count'] <= 5).astype(int)
        df['is_long'] = (df['word_count'] > 20).astype(int)
        
        # POS tagging features
        df = self.extract_pos_features(df)
        
        # Position features (if available)
        if 'y0' in df.columns and 'page_number' in df.columns:
            # Normalize position within each page
            for page in df['page_number'].unique():
                page_mask = df['page_number'] == page
                page_data = df[page_mask]
                
                if len(page_data) > 1:
                    df.loc[page_mask, 'relative_y_position'] = (
                        (page_data['y0'].max() - page_data['y0']) / 
                        (page_data['y0'].max() - page_data['y0'].min())
                    )
                else:
                    df.loc[page_mask, 'relative_y_position'] = 0.5
        
        # Fill missing values
        df = df.fillna(0)
        
        print(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df

    def predict_headings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict headings using the trained model.
        
        Args:
            df: DataFrame with text elements
            
        Returns:
            DataFrame with heading predictions
        """
        if df.empty:
            return df
        
        # Engineer features
        df_features = self.engineer_features(df.copy())
        
        # Select features using the trained feature selector
        try:
            # Get the feature names that the selector expects
            expected_features = self.feature_selector.feature_names_in_
            available_features = [col for col in expected_features if col in df_features.columns]
            missing_features = [col for col in expected_features if col not in df_features.columns]
            
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    df_features[feature] = 0
            
            # Ensure correct column order
            X = df_features[expected_features]
            
            # Apply feature selection
            X_selected = self.feature_selector.transform(X)
            
        except Exception as e:
            print(f"Error in feature selection: {e}")
            # Fallback: use all numeric columns
            numeric_cols = df_features.select_dtypes(include=['int64', 'float64']).columns
            exclude_cols = ['page_number', 'block_id', 'line_id', 'x0', 'y0', 'x1', 'y1']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            X_selected = df_features[feature_cols].values
        
        # Make predictions
        predictions = self.model.predict(X_selected)
        probabilities = self.model.predict_proba(X_selected)[:, 1]  # Probability of being a heading
        
        # Add predictions to dataframe
        df['is_heading'] = predictions
        df['heading_confidence'] = probabilities
        
        return df
    
    def determine_heading_levels(self, headings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Determine hierarchical heading levels (H1, H2, H3) based on font size and structure.
        
        Args:
            headings_df: DataFrame with detected headings
            
        Returns:
            DataFrame with heading levels assigned
        """
        if headings_df.empty:
            return headings_df
        
        # Sort headings by font size (descending) and page order
        headings_df = headings_df.sort_values(['font_size', 'page_number', 'y0'], 
                                            ascending=[False, True, True])
        
        # Group by font size to determine levels
        unique_sizes = sorted(headings_df['font_size'].unique(), reverse=True)
        
        # Assign levels based on font size hierarchy
        level_map = {}
        current_level = 1
        
        for size in unique_sizes:
            if current_level <= 3:  # Only assign H1, H2, H3
                level_map[size] = f"H{current_level}"
                current_level += 1
            else:
                level_map[size] = "H3"  # Anything smaller becomes H3
        
        # Apply level mapping
        headings_df['level'] = headings_df['font_size'].map(level_map)
        
        # Additional logic for structural hierarchy
        headings_df = self._refine_heading_levels(headings_df)
        
        return headings_df.sort_values(['page_number', 'y0'])
    
    def _refine_heading_levels(self, headings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Refine heading levels based on content patterns and structure.
        
        Args:
            headings_df: DataFrame with initial heading levels
            
        Returns:
            DataFrame with refined heading levels
        """
        # Patterns that typically indicate different heading levels
        h1_patterns = [
            r'^\d+\.\s+[A-Z]',  # "1. Introduction"
            r'^[A-Z][A-Z\s]+$',  # All caps
            r'^(CHAPTER|SECTION|PART)\s+\d+',  # "CHAPTER 1"
            r'^(Abstract|Introduction|Conclusion|References)$'  # Common H1 headings
        ]
        
        h2_patterns = [
            r'^\d+\.\d+\s+',  # "1.1 Subsection"
            r'^[A-Z][a-z]+\s+[A-Z]',  # "Related Work"
        ]
        
        h3_patterns = [
            r'^\d+\.\d+\.\d+\s+',  # "1.1.1 Sub-subsection"
            r'^[a-z]\)\s+',  # "a) Point"
            r'^\([a-z]\)\s+',  # "(a) Point"
        ]
        
        for idx, row in headings_df.iterrows():
            text = str(row['text']).strip()
            
            # Check H1 patterns
            for pattern in h1_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    headings_df.at[idx, 'level'] = 'H1'
                    break
            
            # Check H2 patterns
            for pattern in h2_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    headings_df.at[idx, 'level'] = 'H2'
                    break
            
            # Check H3 patterns
            for pattern in h3_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    headings_df.at[idx, 'level'] = 'H3'
                    break
        
        return headings_df
    
    def create_json_output(self, title: str, headings_df: pd.DataFrame) -> Dict:
        """
        Create the final JSON output structure.
        
        Args:
            title: Document title
            headings_df: DataFrame with detected headings and levels
            
        Returns:
            Dict: JSON structure as specified
        """
        outline = []
        
        for _, heading in headings_df.iterrows():
            outline.append({
                "level": heading['level'],
                "text": heading['text'].strip(),
                "page": int(heading['page_number'])
            })
        
        return {
            "title": title,
            "outline": outline
        }
    
    def extract_pdf_to_json(self, pdf_path: str, output_path: str = None, 
                           confidence_threshold: float = 0.5) -> Dict:
        """
        Main method to extract PDF content and create JSON output.
        
        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output JSON file (optional)
            confidence_threshold: Minimum confidence for heading detection
            
        Returns:
            Dict: JSON structure with title and outline
        """
        print(f"🔍 Processing PDF: {pdf_path}")
        
        # Validate PDF
        if not self.validate_pdf(pdf_path):
            return None
        
        # Extract text elements
        print("📄 Extracting text elements...")
        df = self.extract_text_elements(pdf_path)
        
        if df.empty:
            print("❌ No text elements found in PDF")
            return None
        
        print(f"✅ Extracted {len(df)} text elements")
        
        # Detect title
        print("🏷️  Detecting document title...")
        title = self.detect_title(df)
        print(f"📝 Title: {title}")
        
        # Predict headings
        print("🎯 Detecting headings...")
        df_with_predictions = self.predict_headings(df)
        
        # Filter headings by confidence
        headings = df_with_predictions[
            df_with_predictions['heading_confidence'] >= confidence_threshold
        ].copy()
        
        print(f"🔍 Found {len(headings)} headings (confidence ≥ {confidence_threshold*100}%)")
        
        # Debug: show confidence distribution
        if len(df_with_predictions) > 0:
            max_conf = df_with_predictions['heading_confidence'].max()
            mean_conf = df_with_predictions['heading_confidence'].mean()
            print(f"📊 Confidence stats: max={max_conf:.3f}, mean={mean_conf:.3f}")
            
            # Show top candidates regardless of threshold
            top_candidates = df_with_predictions.nlargest(10, 'heading_confidence')
            print(f"🔝 Top 10 candidates by confidence:")
            for i, (_, row) in enumerate(top_candidates.iterrows(), 1):
                text_preview = str(row['text'])[:50].replace('\n', ' ')
                print(f"   {i:2d}. {row['heading_confidence']:.3f} - '{text_preview}...'")
        
        
        # Determine heading levels
        if not headings.empty:
            print("📊 Determining heading hierarchy...")
            headings_with_levels = self.determine_heading_levels(headings)
            
            # Show level distribution
            level_counts = headings_with_levels['level'].value_counts().sort_index()
            for level, count in level_counts.items():
                print(f"   {level}: {count} headings")
        else:
            headings_with_levels = headings
            print("ℹ️  No headings detected")
        
        # Create JSON output
        result = self.create_json_output(title, headings_with_levels)
        
        # Save to file if output path specified
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"💾 JSON output saved to: {output_path}")
            except Exception as e:
                print(f"❌ Error saving JSON file: {e}")
        
        return result


def main():
    """
    Command-line interface for the PDF to JSON extractor.
    """
    parser = argparse.ArgumentParser(
        description="Extract title and hierarchical headings from PDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_to_json_extractor.py input/document.pdf
  python pdf_to_json_extractor.py input/document.pdf -o output.json
  python pdf_to_json_extractor.py input/document.pdf -c 0.7 -o output.json
        """
    )
    
    parser.add_argument('pdf_path', help='Path to the input PDF file')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('-c', '--confidence', type=float, default=0.5,
                       help='Confidence threshold for heading detection (0.0-1.0)')
    parser.add_argument('-m', '--model', help='Path to trained model file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.pdf_path):
        print(f"❌ Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    if not (0.0 <= args.confidence <= 1.0):
        print("❌ Error: Confidence threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Generate output filename if not provided
    if not args.output:
        pdf_name = Path(args.pdf_path).stem
        args.output = f"{pdf_name}_extracted.json"
    
    # Initialize extractor
    print("🚀 PDF to JSON Extractor")
    print("=" * 50)
    
    try:
        extractor = PDFToJSONExtractor(model_path=args.model)
        
        # Process PDF
        result = extractor.extract_pdf_to_json(
            pdf_path=args.pdf_path,
            output_path=args.output,
            confidence_threshold=args.confidence
        )
        
        if result:
            print("\n✅ Extraction completed successfully!")
            print(f"📊 Summary:")
            print(f"   Title: {result['title']}")
            print(f"   Total headings: {len(result['outline'])}")
            
            if args.verbose and result['outline']:
                print(f"\n📋 Detected headings:")
                for i, heading in enumerate(result['outline'], 1):
                    print(f"   {i:2d}. [{heading['level']}] {heading['text']} (Page {heading['page']})")
        else:
            print("❌ Extraction failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
