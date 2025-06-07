# -*- coding: utf-8 -*-
"""
Debug script to identify specific differences between Khmer syllable segmentation methods.
"""

import sys
from subword_cluster import (
    khmer_syllables,
    khmer_syllables_no_regex,
    khmer_syllables_no_regex_fast, 
    khmer_syllables_advanced,
)

def debug_character_handling(text):
    """Debug how each function handles individual characters and character types"""
    print(f"=== Character Analysis: '{text}' ===")
    
    # Show character codes
    print("Character codes:")
    for i, char in enumerate(text):
        print(f"  {i}: '{char}' -> U+{ord(char):04X}")
    
    print("\nMethod results:")
    methods = [
        ('regex_basic', khmer_syllables),
        ('regex_advanced', khmer_syllables_advanced),
        ('non_regex_original', khmer_syllables_no_regex),
        ('non_regex_fast', khmer_syllables_no_regex_fast),
    ]
    
    for name, func in methods:
        try:
            result = func(text)
            print(f"  {name}: {result} (count: {len(result)})")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    
    print("-" * 50)

def test_specific_cases():
    """Test specific problematic cases"""
    
    # Test cases based on the debug report
    test_cases = [
        "Apple",  # English word
        "Thon",   # English word  
        "២០១៦", # Khmer numbers
        "។",      # Khmer punctuation
        "\u200b", # Zero-width space
        "អ្នក",   # Simple Khmer
        "ឲ្យ",    # Khmer with subscript
    ]
    
    for case in test_cases:
        debug_character_handling(case)

def compare_sample_text():
    """Compare methods on a sample text that shows differences"""
    sample = "Apple ឲ្យប្រើអត់យកលុយ"
    
    print(f"=== Comparing Sample Text ===")
    print(f"Text: {sample}")
    print(f"Length: {len(sample)}")
    
    methods = [
        ('regex_basic', khmer_syllables),
        ('regex_advanced', khmer_syllables_advanced),
        ('non_regex_original', khmer_syllables_no_regex),
        ('non_regex_fast', khmer_syllables_no_regex_fast),
    ]
    
    results = {}
    for name, func in methods:
        try:
            result = func(sample)
            results[name] = result
            print(f"\n{name}:")
            print(f"  Result: {result}")
            print(f"  Count: {len(result)}")
            print(f"  Join: {'|'.join(result)}")
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
    
    # Find differences
    print("\n=== Differences Analysis ===")
    if len(set(len(r) for r in results.values())) > 1:
        print("⚠️  Different syllable counts detected!")
        for name, result in results.items():
            print(f"  {name}: {len(result)} syllables")
    
    # Compare syllable by syllable
    max_len = max(len(r) for r in results.values())
    print(f"\nSyllable-by-syllable comparison (max length: {max_len}):")
    
    for i in range(max_len):
        print(f"  Position {i}:")
        syllables_at_pos = {}
        for name, result in results.items():
            if i < len(result):
                syllable = result[i]
                syllables_at_pos[name] = syllable
                print(f"    {name}: '{syllable}' (U+{ord(syllable[0]):04X} if single char)")
            else:
                print(f"    {name}: <none>")
        
        # Check if all syllables at this position are the same
        unique_syllables = set(syllables_at_pos.values())
        if len(unique_syllables) > 1:
            print(f"    ⚠️  DIFFERENCE at position {i}!")

if __name__ == "__main__":
    print("=== Khmer Syllable Segmentation Debug ===\n")
    
    # Test specific cases
    test_specific_cases()
    
    # Compare a sample text
    compare_sample_text() 