#!/usr/bin/env python3
"""
Real Dataset Creator for PDF Heading Detection
Creates labeled training data from PDF documents
"""

import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from collections import defaultdict
import ssl

# Handle SSL certificate issues for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk import pos_tag, word_tokenize


class DatasetCreator:
    """Creates labeled datasets for training PDF heading detection models"""
    
    def __init__(self):
        self.labeled_data = []
        
    def extract_text_elements(self, pdf_path):
        """Extract all text elements from a PDF with their properties"""
        print(f"Extracting text elements from {pdf_path}...")
        doc = fitz.open(pdf_path)
        elements = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            # Calculate document-level font threshold
            font_sizes = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
            
            font_threshold = max(set(font_sizes), key=font_sizes.count) if font_sizes else 12
            
            for block_idx, block in enumerate(blocks):
                if "lines" in block:
                    for line_idx, line in enumerate(block["lines"]):
                        # Merge spans in same line
                        line_text = " ".join([span["text"].strip() for span in line["spans"]])
                        if not line_text or len(line_text.strip()) == 0:
                            continue
                        
                        # Calculate average properties for the line
                        avg_font_size = np.mean([span["size"] for span in line["spans"]])
                        is_bold = any(bool(span["flags"] & 16) for span in line["spans"])
                        is_italic = any(bool(span["flags"] & 2) for span in line["spans"])
                        
                        # Extract bounding box for position features
                        y0 = min(span["bbox"][1] for span in line["spans"])
                        x0 = min(span["bbox"][0] for span in line["spans"])
                        x1 = max(span["bbox"][2] for span in line["spans"])
                        y1 = max(span["bbox"][3] for span in line["spans"])
                        
                        # Font family/name
                        font_name = line["spans"][0]["font"] if line["spans"] else "unknown"
                        
                        element = {
                            'pdf_file': Path(pdf_path).name,
                            'page_number': page_num + 1,
                            'block_id': block_idx,
                            'line_id': line_idx,
                            'text': line_text,
                            'font_size': avg_font_size,
                            'font_threshold': font_threshold,
                            'is_bold': is_bold,
                            'is_italic': is_italic,
                            'font_name': font_name,
                            'x0': x0,
                            'y0': y0,
                            'x1': x1,
                            'y1': y1,
                            'width': x1 - x0,
                            'height': y1 - y0,
                            # To be labeled
                            'is_heading': None,
                            'heading_level': None
                        }
                        elements.append(element)
        
        doc.close()
        return elements
    
    def auto_label_heuristics(self, elements):
        """Apply sophisticated heuristics to automatically label obvious cases"""
        labeled_elements = []
        
        for element in elements:
            text = element['text'].strip()
            font_size = element['font_size']
            font_threshold = element['font_threshold']
            is_bold = element['is_bold']
            
            # Initialize labels
            is_heading = 0
            heading_level = 0
            confidence = 0.0
            
            # Rule 1: Numbered sections (high confidence)
            if re.match(r'^\s*\d+\.?\s+[A-Z]', text) and len(text.split()) <= 15:
                is_heading = 1
                heading_level = 1
                confidence = 0.9
            
            # Rule 2: Subsections (medium confidence)
            elif re.match(r'^\s*\d+\.\d+\.?\s+[A-Z]', text) and len(text.split()) <= 12:
                is_heading = 1
                heading_level = 2
                confidence = 0.85
            
            # Rule 3: Sub-subsections
            elif re.match(r'^\s*\d+\.\d+\.\d+\.?\s+[A-Z]', text) and len(text.split()) <= 10:
                is_heading = 1
                heading_level = 3
                confidence = 0.8
            
            # Rule 4: All caps short text
            elif text.isupper() and len(text.split()) <= 8 and font_size >= font_threshold:
                is_heading = 1
                heading_level = 1
                confidence = 0.7
            
            # Rule 5: Bold + larger font + title case
            elif (is_bold and 
                  font_size > font_threshold * 1.1 and 
                  text.istitle() and 
                  len(text.split()) <= 10):
                is_heading = 1
                heading_level = 1 if font_size > font_threshold * 1.3 else 2
                confidence = 0.8
            
            # Rule 6: Common heading patterns
            elif re.match(r'^\s*(abstract|introduction|conclusion|references|bibliography|acknowledgments|appendix)', text.lower()):
                is_heading = 1
                heading_level = 1
                confidence = 0.9
            
            # Rule 7: Roman numerals
            elif re.match(r'^\s*[IVX]+\.?\s+[A-Z]', text) and len(text.split()) <= 8:
                is_heading = 1
                heading_level = 1
                confidence = 0.8
            
            # Rule 8: Lettered sections
            elif re.match(r'^\s*[A-Z]\.?\s+[A-Z]', text) and len(text.split()) <= 8:
                is_heading = 1
                heading_level = 2
                confidence = 0.75
            
            # Rule 9: Very large font (likely title)
            elif font_size > font_threshold * 1.5 and len(text.split()) <= 12:
                is_heading = 1
                heading_level = 1
                confidence = 0.85
            
            # Add computed features
            element.update({
                'is_heading': is_heading,
                'heading_level': heading_level,
                'auto_confidence': confidence,
                'relative_font_size': font_size / font_threshold if font_threshold > 0 else 1,
                'word_count': len(text.split()),
                'char_count': len(text),
                'starts_with_number': int(bool(re.match(r'^\s*\d+', text))),
                'ends_with_colon': int(text.endswith(':')),
                'is_all_caps': int(text.isupper()),
                'is_title_case': int(text.istitle()),
                'has_punctuation': int(any(c in text for c in '.,;!?')),
            })
            
            labeled_elements.append(element)
        
        return labeled_elements
    
    def create_interactive_labeling_interface(self, elements, output_file):
        """Create an interactive interface for manual labeling"""
        print(f"\n{'='*80}")
        print("INTERACTIVE LABELING INTERFACE")
        print('='*80)
        print("Instructions:")
        print("- Review each text element and classify as heading or not")
        print("- For headings, specify level (1=main, 2=sub, 3=subsub, etc.)")
        print("- Auto-suggestions are provided based on heuristics")
        print("- Commands: 'y'=heading, 'n'=not heading, 's'=skip, 'q'=quit, 'save'=save progress")
        print("- For headings, you can specify level: 'y1', 'y2', 'y3', etc.")
        print("-" * 80)
        
        labeled_count = 0
        
        for i, element in enumerate(elements):
            if element['auto_confidence'] > 0.8:
                # Auto-accept high confidence predictions
                labeled_count += 1
                continue
                
            print(f"\n[{i+1}/{len(elements)}] Page {element['page_number']}")
            print(f"Text: {element['text'][:100]}{'...' if len(element['text']) > 100 else ''}")
            print(f"Font: {element['font_size']:.1f} | Bold: {element['is_bold']} | Relative Size: {element['relative_font_size']:.2f}")
            
            if element['auto_confidence'] > 0:
                suggestion = f"Suggested: {'Heading' if element['is_heading'] else 'Not Heading'}"
                if element['is_heading']:
                    suggestion += f" (Level {element['heading_level']})"
                suggestion += f" [Confidence: {element['auto_confidence']:.1%}]"
                print(f"🤖 {suggestion}")
            
            while True:
                response = input("Label (y/n/s/q/save): ").strip().lower()
                
                if response == 'q':
                    print("Quitting labeling session...")
                    return self.save_dataset(elements[:i], output_file)
                
                elif response == 'save':
                    self.save_dataset(elements[:i], output_file)
                    print(f"Progress saved. {labeled_count} elements labeled so far.")
                    continue
                
                elif response == 's':
                    break
                
                elif response == 'n':
                    element['is_heading'] = 0
                    element['heading_level'] = 0
                    element['manual_label'] = True
                    labeled_count += 1
                    break
                
                elif response.startswith('y'):
                    element['is_heading'] = 1
                    element['manual_label'] = True
                    
                    # Extract level if specified
                    if len(response) > 1 and response[1:].isdigit():
                        element['heading_level'] = int(response[1:])
                    else:
                        # Ask for level
                        while True:
                            level_input = input("Heading level (1-5): ").strip()
                            if level_input.isdigit() and 1 <= int(level_input) <= 5:
                                element['heading_level'] = int(level_input)
                                break
                            print("Please enter a number between 1 and 5")
                    
                    labeled_count += 1
                    break
                
                else:
                    print("Invalid input. Use y/n/s/q/save")
        
        print(f"\nLabeling complete! {labeled_count} elements labeled.")
        return self.save_dataset(elements, output_file)
    
    def save_dataset(self, elements, output_file):
        """Save the labeled dataset"""
        df = pd.DataFrame(elements)
        df.to_csv(output_file, index=False)
        
        # Also save as JSON for backup
        json_file = output_file.replace('.csv', '.json')
        with open(json_file, 'w') as f:
            json.dump(elements, f, indent=2)
        
        print(f"Dataset saved to: {output_file}")
        print(f"Backup saved to: {json_file}")
        
        # Print statistics
        total_elements = len(elements)
        headings = sum(1 for e in elements if e['is_heading'] == 1)
        auto_labeled = sum(1 for e in elements if e.get('auto_confidence', 0) > 0.8)
        manual_labeled = sum(1 for e in elements if e.get('manual_label', False))
        
        print(f"\nDataset Statistics:")
        print(f"Total elements: {total_elements}")
        if total_elements > 0:
            print(f"Headings: {headings} ({headings/total_elements:.1%})")
            print(f"Non-headings: {total_elements - headings} ({(total_elements - headings)/total_elements:.1%})")
        else:
            print(f"Headings: {headings}")
            print(f"Non-headings: {total_elements - headings}")
        print(f"Auto-labeled: {auto_labeled}")
        print(f"Manually labeled: {manual_labeled}")
        
        return df
    
    def load_existing_dataset(self, dataset_file):
        """Load an existing dataset for continuation"""
        try:
            if dataset_file.endswith('.json'):
                with open(dataset_file, 'r') as f:
                    elements = json.load(f)
            else:
                df = pd.read_csv(dataset_file)
                elements = df.to_dict('records')
            
            print(f"Loaded existing dataset with {len(elements)} elements")
            return elements
        except FileNotFoundError:
            print(f"Dataset file {dataset_file} not found")
            return []
    
    def create_comprehensive_dataset(self, pdf_folder, output_file=None, interactive=True):
        """Create a comprehensive labeled dataset from multiple PDFs"""
        pdf_folder = Path(pdf_folder)
        pdf_files = list(pdf_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_folder}")
            return None
        
        if output_file is None:
            output_file = pdf_folder / "heading_detection_dataset.csv"
        
        all_elements = []
        
        # Extract elements from all PDFs
        for pdf_file in pdf_files:
            try:
                elements = self.extract_text_elements(str(pdf_file))
                all_elements.extend(elements)
                print(f"Extracted {len(elements)} elements from {pdf_file.name}")
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {str(e)}")
        
        if not all_elements:
            print("No text elements extracted from PDFs")
            return None
        
        print(f"\nTotal elements extracted: {len(all_elements)}")
        
        # Apply auto-labeling heuristics
        print("Applying automatic labeling heuristics...")
        labeled_elements = self.auto_label_heuristics(all_elements)
        
        # Count auto-labeled elements
        high_confidence_count = sum(1 for e in labeled_elements if e['auto_confidence'] > 0.8)
        medium_confidence_count = sum(1 for e in labeled_elements if 0.5 < e['auto_confidence'] <= 0.8)
        low_confidence_count = sum(1 for e in labeled_elements if 0 < e['auto_confidence'] <= 0.5)
        unlabeled_count = sum(1 for e in labeled_elements if e['auto_confidence'] == 0)
        
        print(f"\nAuto-labeling results:")
        print(f"High confidence (>80%): {high_confidence_count}")
        print(f"Medium confidence (50-80%): {medium_confidence_count}")
        print(f"Low confidence (0-50%): {low_confidence_count}")
        print(f"Unlabeled: {unlabeled_count}")
        
        if interactive and (medium_confidence_count + low_confidence_count + unlabeled_count) > 0:
            print(f"\nStarting interactive labeling for {medium_confidence_count + low_confidence_count + unlabeled_count} uncertain elements...")
            return self.create_interactive_labeling_interface(labeled_elements, output_file)
        else:
            return self.save_dataset(labeled_elements, output_file)
    
    def augment_dataset_with_synthetic_examples(self, dataset_file):
        """Augment the dataset with synthetic examples based on patterns"""
        df = pd.read_csv(dataset_file)
        
        # Analyze patterns in existing headings
        headings = df[df['is_heading'] == 1]
        
        if len(headings) == 0:
            print("No headings found in dataset for augmentation")
            return df
        
        print(f"Analyzing {len(headings)} headings for pattern extraction...")
        
        synthetic_examples = []
        
        # Pattern 1: Numbered sections
        for i in range(1, 10):
            for topic in ['Introduction', 'Background', 'Methodology', 'Results', 'Discussion', 'Conclusion']:
                synthetic_examples.append({
                    'pdf_file': 'synthetic',
                    'page_number': 1,
                    'text': f"{i}. {topic}",
                    'font_size': 12.0,
                    'font_threshold': 10.0,
                    'is_bold': True,
                    'is_heading': 1,
                    'heading_level': 1,
                    'relative_font_size': 1.2,
                    'word_count': 2,
                    'is_synthetic': True
                })
        
        # Pattern 2: Subsections
        for i in range(1, 5):
            for j in range(1, 4):
                for topic in ['Overview', 'Analysis', 'Implementation', 'Evaluation']:
                    synthetic_examples.append({
                        'pdf_file': 'synthetic',
                        'page_number': 1,
                        'text': f"{i}.{j} {topic}",
                        'font_size': 11.0,
                        'font_threshold': 10.0,
                        'is_bold': True,
                        'is_heading': 1,
                        'heading_level': 2,
                        'relative_font_size': 1.1,
                        'word_count': 2,
                        'is_synthetic': True
                    })
        
        # Add synthetic examples to dataset
        synthetic_df = pd.DataFrame(synthetic_examples)
        augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
        
        # Save augmented dataset
        augmented_file = dataset_file.replace('.csv', '_augmented.csv')
        augmented_df.to_csv(augmented_file, index=False)
        
        print(f"Added {len(synthetic_examples)} synthetic examples")
        print(f"Augmented dataset saved to: {augmented_file}")
        
        return augmented_df


def main():
    """Main function for dataset creation"""
    print("PDF Heading Detection Dataset Creator")
    print("=" * 50)
    
    creator = DatasetCreator()
    
    # Create dataset from input folder
    input_folder = "/Users/talhaansari/Developer/Adobe/1a new/input"
    dataset_file = "/Users/talhaansari/Developer/Adobe/1a new/training_dataset.csv"
    
    print(f"Creating dataset from PDFs in: {input_folder}")
    
    # Ask user for labeling preference
    print("\nDataset creation options:")
    print("1. Automatic labeling only (fast, good for initial training)")
    print("2. Interactive labeling (slower, more accurate)")
    
    while True:
        choice = input("Choose option (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    interactive = (choice == '2')
    
    # Create the dataset
    dataset_df = creator.create_comprehensive_dataset(
        input_folder, 
        dataset_file, 
        interactive=interactive
    )
    
    if dataset_df is not None:
        print(f"\n✅ Dataset created successfully!")
        
        # Ask if user wants to augment with synthetic data
        augment = input("\nAdd synthetic examples to improve training? (y/n): ").strip().lower()
        if augment == 'y':
            creator.augment_dataset_with_synthetic_examples(dataset_file)
        
        print(f"\n📊 Dataset ready for training at: {dataset_file}")
    else:
        print("❌ Failed to create dataset")


if __name__ == "__main__":
    main()
