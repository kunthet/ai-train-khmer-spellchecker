#!/usr/bin/env python3
"""
Debug specific inconsistencies found in the kmwiki analysis.
This script will help identify the exact differences between methods.
"""

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

def analyze_specific_file(filename):
    """Analyze a specific file that showed inconsistencies."""
    
    filepath = Path(r"D:\data\ML\KhmerText\seg_kmwiki_data\kmwiki_data") / filename
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    print(f"🔍 ANALYZING: {filename}")
    print("=" * 60)
    
    # Read and reconstruct text
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    reconstructed = reconstruct_text(content)
    
    print(f"Original length: {len(content):,} characters")
    print(f"Reconstructed length: {len(reconstructed):,} characters")
    print(f"Sample reconstructed: {repr(reconstructed[:100])}...")
    print()
    
    # Test all methods
    methods = {
        'khmer_syllables': khmer_syllables,
        'khmer_syllables_advanced': khmer_syllables_advanced,
        'khmer_syllables_no_regex': khmer_syllables_no_regex,
        'khmer_syllables_no_regex_fast': khmer_syllables_no_regex_fast
    }
    
    results = {}
    for name, func in methods.items():
        result = func(reconstructed)
        results[name] = result
        print(f"{name:25}: {len(result):,} tokens")
    
    print()
    
    # Find differences
    result_lists = list(results.values())
    if not all(r == result_lists[0] for r in result_lists):
        print("🔍 DETAILED DIFFERENCE ANALYSIS:")
        print("-" * 40)
        
        # Group by identical results
        groups = {}
        for name, result in results.items():
            result_key = str(result)
            if result_key not in groups:
                groups[result_key] = []
            groups[result_key].append((name, result))
        
        print(f"Found {len(groups)} different result patterns:")
        print()
        
        for i, (result_key, group) in enumerate(groups.items(), 1):
            method_names = [name for name, _ in group]
            result = group[0][1]
            
            print(f"GROUP {i}: {', '.join(method_names)}")
            print(f"  Length: {len(result):,} tokens")
            print(f"  First 15: {result[:15]}")
            print(f"  Last 15:  {result[-15:]}")
            print()
        
        # Find where they first differ
        min_length = min(len(r) for r in result_lists)
        print(f"Comparing first {min_length:,} tokens to find differences...")
        
        first_diff_pos = None
        for i in range(min_length):
            tokens_at_pos = [r[i] for r in result_lists]
            if not all(token == tokens_at_pos[0] for token in tokens_at_pos):
                first_diff_pos = i
                break
        
        if first_diff_pos is not None:
            print(f"📍 First difference at position {first_diff_pos}:")
            for name, result in results.items():
                if first_diff_pos < len(result):
                    print(f"  {name:25}: {repr(result[first_diff_pos])}")
                else:
                    print(f"  {name:25}: [END OF LIST]")
            print()
            
            # Show context around first difference
            context_start = max(0, first_diff_pos - 5)
            context_end = min(min_length, first_diff_pos + 10)
            
            print(f"📍 Context around position {first_diff_pos} (showing {context_start}-{context_end}):")
            for name, result in results.items():
                context = result[context_start:context_end]
                print(f"  {name:25}: {context}")
            print()
    else:
        print("✅ All methods produce identical results!")

def main():
    """Test a few problematic files identified from the log."""
    
    problematic_files = [
        "Computer.txt",
        "Ascaridia Galli.txt", 
        "Baba Vanga.txt",
        "Film edit.txt",
        "Gloomy Sunday.txt",
        "Harvard University.txt",
        "Nine Steps To a Strategies Marketing Plan.txt"
    ]
    
    for filename in problematic_files:  # Test all files
        analyze_specific_file(filename)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 