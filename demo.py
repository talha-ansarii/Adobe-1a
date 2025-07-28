#!/usr/bin/env python3
"""
Demo script showing different ways to use the PDF heading extractor
"""

import os
import json
from pathlib import Path

def main():
    print("🚀 PDF to JSON Extractor - Demo")
    print("=" * 50)
    
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    print("\n📋 Available demo commands:")
    print("1. Extract single PDF with default settings")
    print("2. Extract with custom confidence threshold")
    print("3. Extract with verbose output")
    print("4. Process all PDFs with different thresholds")
    
    demos = [
        {
            "name": "Basic extraction (example.pdf)",
            "command": "python pdf_to_json_extractor.py input/example.pdf -c 0.4 -o demo_example.json"
        },
        {
            "name": "High confidence extraction (Kehe 103.pdf)",
            "command": "python pdf_to_json_extractor.py 'input/Kehe 103.pdf' -c 0.5 -o demo_kehe.json"
        },
        {
            "name": "Verbose extraction with low confidence",
            "command": "python pdf_to_json_extractor.py input/EJ1172284.pdf -c 0.3 -o demo_research.json --verbose"
        }
    ]
    
    print("\n🎯 Demo Commands:")
    for i, demo in enumerate(demos, 1):
        print(f"\n{i}. {demo['name']}")
        print(f"   Command: {demo['command']}")
    
    print("\n📊 Expected JSON Output Format:")
    example_json = {
        "title": "Understanding AI",
        "outline": [
            {"level": "H1", "text": "Introduction", "page": 1},
            {"level": "H2", "text": "What is AI?", "page": 2},
            {"level": "H3", "text": "History of AI", "page": 3}
        ]
    }
    print(json.dumps(example_json, indent=2))
    
    print("\n💡 Tips:")
    print("- Use confidence threshold 0.3-0.6 for most documents")
    print("- Lower threshold = more headings (may include false positives)")
    print("- Higher threshold = fewer headings (may miss some)")
    print("- Use --verbose flag for debugging")
    
    print("\n🔧 Quick Test:")
    print("To test the system right now, run:")
    print("python pdf_to_json_extractor.py input/example.pdf -o test_output.json")

if __name__ == "__main__":
    main()
