#!/usr/bin/env python3
"""Simple test to verify consistency fixes"""

from pathlib import Path
from subword_cluster import (
    khmer_syllables,
    khmer_syllables_advanced,
    khmer_syllables_no_regex,
    khmer_syllables_no_regex_fast
)

def reconstruct_text(tokenized_text):
    """Reconstruct untokenized text by removing single spaces but keeping double spaces."""
    placeholder = "<<<DOUBLE_SPACE>>>"
    text = tokenized_text.replace("  ", placeholder)
    text = text.replace(" ", "")
    text = text.replace(placeholder, "  ")
    return text

def test_files():
    """Test problematic files to verify consistency."""
    
    test_files = [
        "Computer.txt",
        "Ascaridia Galli.txt", 
        "Baba Vanga.txt",
        "Film edit.txt",
        "Gloomy Sunday.txt"
    ]
    
    methods = [
        khmer_syllables,
        khmer_syllables_advanced,
        khmer_syllables_no_regex,
        khmer_syllables_no_regex_fast
    ]
    
    base_path = Path(r"D:\data\ML\KhmerText\seg_kmwiki_data\kmwiki_data")
    
    results_summary = []
    
    for filename in test_files:
        filepath = base_path / filename
        
        if not filepath.exists():
            print(f"SKIP: {filename} - file not found")
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            reconstructed = reconstruct_text(content)
            
            # Test all methods
            results = []
            for method in methods:
                result = method(reconstructed)
                results.append(len(result))
            
            # Check consistency
            all_same = len(set(results)) == 1
            
            status = "CONSISTENT" if all_same else "INCONSISTENT"
            print(f"{status}: {filename} - lengths: {results}")
            results_summary.append((filename, all_same, results))
            
        except Exception as e:
            print(f"ERROR: {filename} - {e}")
    
    print("\nSUMMARY:")
    print("-" * 50)
    consistent_count = sum(1 for _, is_consistent, _ in results_summary if is_consistent)
    total_count = len(results_summary)
    
    print(f"Consistent files: {consistent_count}/{total_count}")
    
    if consistent_count == total_count:
        print("SUCCESS: All test files are now consistent!")
    else:
        print("ATTENTION: Some files still show inconsistencies")
        for filename, is_consistent, results in results_summary:
            if not is_consistent:
                print(f"  - {filename}: {results}")

if __name__ == "__main__":
    test_files() 