#!/usr/bin/env python3
"""
Verification script for PDF heading detection results
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(input_folder):
    """Analyze the heading detection results"""
    input_path = Path(input_folder)
    
    # Find all result CSV files
    result_files = list(input_path.glob("*_heading_analysis.csv"))
    
    print("="*80)
    print("DETAILED ANALYSIS OF PDF HEADING DETECTION RESULTS")
    print("="*80)
    
    for result_file in result_files:
        pdf_name = result_file.stem.replace("_heading_analysis", "")
        print(f"\n📄 Analysis for: {pdf_name}.pdf")
        print("-" * 50)
        
        try:
            df = pd.read_csv(result_file)
            
            # Basic statistics
            total_elements = len(df)
            if 'predicted_heading' in df.columns:
                detected_headings = df[df['predicted_heading'] == 1]
                num_headings = len(detected_headings)
                
                print(f"Total text elements: {total_elements}")
                print(f"Detected headings: {num_headings}")
                print(f"Heading ratio: {num_headings/total_elements:.2%}")
                
                if num_headings > 0:
                    # Show feature analysis
                    print(f"\n🎯 Detected Headings:")
                    for i, (_, row) in enumerate(detected_headings.iterrows(), 1):
                        confidence = row.get('heading_probability', 0) * 100
                        print(f"{i:2d}. {row['text'][:80]}{'...' if len(row['text']) > 80 else ''}")
                        print(f"    📊 Confidence: {confidence:.1f}% | Font: {row['font_size']:.1f} | Bold: {'Yes' if row['is_bold'] else 'No'}")
                    
                    # Feature statistics
                    print(f"\n📈 Feature Analysis:")
                    print(f"Average font size (headings): {detected_headings['font_size'].mean():.1f}")
                    print(f"Average font size (body): {df[df['predicted_heading'] == 0]['font_size'].mean():.1f}")
                    print(f"Bold text percentage (headings): {detected_headings['is_bold'].mean():.1%}")
                    print(f"Bold text percentage (body): {df[df['predicted_heading'] == 0]['is_bold'].mean():.1%}")
                    
                    # Word count analysis
                    print(f"Average words (headings): {detected_headings['words'].mean():.1f}")
                    print(f"Average words (body): {df[df['predicted_heading'] == 0]['words'].mean():.1f}")
                    
                else:
                    print("❌ No headings detected")
            else:
                print("❌ No heading predictions available (insufficient training data)")
                print(f"Total text elements: {total_elements}")
                
        except Exception as e:
            print(f"❌ Error reading {result_file}: {str(e)}")
    
    # Create visualization if possible
    create_visualization(input_path, result_files)

def create_visualization(input_path, result_files):
    """Create visualizations of the results"""
    try:
        print(f"\n📊 Creating visualization...")
        
        # Collect data for visualization
        all_data = []
        
        for result_file in result_files:
            pdf_name = result_file.stem.replace("_heading_analysis", "")
            df = pd.read_csv(result_file)
            
            if 'predicted_heading' in df.columns:
                df['pdf_name'] = pdf_name
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('PDF Heading Detection Analysis', fontsize=16)
            
            # 1. Font size distribution
            axes[0, 0].hist(combined_df[combined_df['predicted_heading'] == 1]['font_size'], 
                           alpha=0.7, label='Headings', bins=20, color='red')
            axes[0, 0].hist(combined_df[combined_df['predicted_heading'] == 0]['font_size'], 
                           alpha=0.7, label='Body Text', bins=20, color='blue')
            axes[0, 0].set_xlabel('Font Size')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Font Size Distribution')
            axes[0, 0].legend()
            
            # 2. Word count distribution
            axes[0, 1].hist(combined_df[combined_df['predicted_heading'] == 1]['words'], 
                           alpha=0.7, label='Headings', bins=20, color='red')
            axes[0, 1].hist(combined_df[combined_df['predicted_heading'] == 0]['words'], 
                           alpha=0.7, label='Body Text', bins=50, color='blue')
            axes[0, 1].set_xlabel('Word Count')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Word Count Distribution')
            axes[0, 1].legend()
            axes[0, 1].set_xlim(0, 50)  # Focus on reasonable range
            
            # 3. Heading count by PDF
            heading_counts = combined_df.groupby('pdf_name')['predicted_heading'].sum()
            axes[1, 0].bar(heading_counts.index, heading_counts.values, color='green')
            axes[1, 0].set_xlabel('PDF Document')
            axes[1, 0].set_ylabel('Number of Headings')
            axes[1, 0].set_title('Headings Detected per Document')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Feature importance (if available)
            if 'heading_probability' in combined_df.columns:
                prob_data = combined_df[combined_df['predicted_heading'] == 1]['heading_probability']
                axes[1, 1].hist(prob_data, bins=20, color='orange', alpha=0.7)
                axes[1, 1].set_xlabel('Heading Probability')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Heading Confidence Distribution')
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = input_path / "heading_detection_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📈 Visualization saved to: {plot_path}")
            
            plt.show()
            
    except Exception as e:
        print(f"❌ Error creating visualization: {str(e)}")

def main():
    """Main verification function"""
    input_folder = "/Users/talhaansari/Developer/Adobe/1a new/input"
    analyze_results(input_folder)

if __name__ == "__main__":
    main()
