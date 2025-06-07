#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from subword_cluster import (
    khmer_syllables,
    khmer_syllables_advanced,
    khmer_syllables_no_regex,
)

def test_zwsp_removal_impact():
    """Test how removing zero-width spaces affects consistency"""
    
    # Sample text with zero-width spaces (common in Khmer web text)
    text_with_zwsp = "Apple\u200bឲ្យ\u200bប្រើ"
    text_without_zwsp = text_with_zwsp.replace('\u200b', '')
    
    methods = [
        ('regex_basic', khmer_syllables),
        ('regex_advanced', khmer_syllables_advanced),
        ('non_regex_original', khmer_syllables_no_regex),
    ]
    
    print("=== BEFORE removing zero-width spaces ===")
    print(f"Text: {repr(text_with_zwsp)}")
    results_before = {}
    for name, func in methods:
        result = func(text_with_zwsp)
        results_before[name] = result
        print(f"{name:18}: {result}")
    
    # Check consistency
    unique_before = set(str(r) for r in results_before.values())
    print(f"Consistent: {'✅ YES' if len(unique_before) == 1 else '❌ NO'}")
    
    print("\n=== AFTER removing zero-width spaces ===")
    print(f"Text: {repr(text_without_zwsp)}")
    results_after = {}
    for name, func in methods:
        result = func(text_without_zwsp)
        results_after[name] = result
        print(f"{name:18}: {result}")
    
    # Check consistency
    unique_after = set(str(r) for r in results_after.values())
    print(f"Consistent: {'✅ YES' if len(unique_after) == 1 else '❌ NO'}")
    
    print("\n=== Analysis ===")
    print("The zero-width spaces act as syllable boundaries.")
    print("When removed, different methods handle the merged text differently!")
    
    if len(unique_before) == 1 and len(unique_after) > 1:
        print("\n❌ REMOVING ZERO-WIDTH SPACES BROKE CONSISTENCY!")
        print("The cleanup code is causing the inconsistencies.")
    elif len(unique_before) > 1 and len(unique_after) == 1:
        print("\n✅ Removing zero-width spaces improved consistency")
    else:
        print(f"\nBoth scenarios: before={len(unique_before)} after={len(unique_after)} unique results")

if __name__ == "__main__":
    test_zwsp_removal_impact() 