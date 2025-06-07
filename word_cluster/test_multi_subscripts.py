# -*- coding: utf-8 -*-
"""
Test script to debug multi-subscript generation
"""

from generate_synthetic_khmer_tokens import KhmerTokenGenerator

def test_multi_subscripts():
    generator = KhmerTokenGenerator()
    
    print("=== Testing Multi-Subscript Generation ===")
    
    # Test double subscript clusters
    print("\n1. Testing double subscript clusters...")
    double_clusters = generator.generate_double_subscript_clusters()
    print(f"Generated {len(double_clusters)} double subscript clusters")
    print("First 10 double clusters:")
    for i, cluster in enumerate(list(double_clusters)[:10]):
        print(f"  {i+1}. {cluster}")
    
    # Test if specific patterns exist
    target_patterns = ['ស្ត្រ', 'ហ្វ្រ', 'ង្ក្រ', 'សាស្ត្រ', 'ស្ត្រី']
    print(f"\n2. Checking for specific target patterns...")
    for pattern in target_patterns:
        if pattern in double_clusters:
            print(f"  ✅ Found: {pattern}")
        else:
            print(f"  ❌ Missing: {pattern}")
    
    # Test double subscript with vowels
    print("\n3. Testing double subscript with vowels...")
    double_vowel_combinations = generator.generate_double_subscript_vowel_combinations()
    print(f"Generated {len(double_vowel_combinations)} double subscript+vowel combinations")
    
    # Check for specific examples
    specific_examples = ['ស្ត្រី', 'សាស្ត្រ', 'សាស្ត្រា', 'ហ្វ្រង្ក', 'អង្ក្រង']
    print(f"\n4. Checking for specific documented examples...")
    for example in specific_examples:
        if example in double_vowel_combinations:
            print(f"  ✅ Found: {example}")
        else:
            print(f"  ❌ Missing: {example}")
    
    # Test documented specific patterns
    print("\n5. Testing documented specific patterns...")
    documented_patterns = generator.generate_documented_specific_patterns()
    print(f"Generated {len(documented_patterns)} documented patterns")
    print("All documented patterns:")
    for pattern in documented_patterns:
        print(f"  {pattern}")
    
    # Test triple subscripts
    print("\n6. Testing triple subscript clusters...")
    triple_clusters = generator.generate_triple_subscript_clusters()
    print(f"Generated {len(triple_clusters)} triple subscript clusters")
    print("Triple clusters:")
    for cluster in triple_clusters:
        print(f"  {cluster}")

if __name__ == "__main__":
    test_multi_subscripts() 