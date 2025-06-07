#!/usr/bin/env python3
"""Test script to verify all segmentation functions produce consistent results"""

from subword_cluster import *

def test_problematic_texts():
    """Test the previously problematic texts"""
    
    # Test texts from the original problems
    test_cases = [
        "វង់ភ្លេងអាយ៉ៃ កើតឡើងដោយបុរសឈ្មោះយ៉ៃ តាំងពីចុងសតវត្សទី១៩មកម្ល៉េះ - វិទ្យុវាយោ - VAYO FM Radioព័ត៌មាន",
        "អាយ៉ៃឆ្លងឆ្លើយ គឺជាការច្រៀងឆ្លងឆ្លើយកំណាព្យកាព្យឃ្លោងដោយឥតព្រាងទុក ។",
        "ជនភៀសខ្លួន ព្យាយាម​ឆ្លងទឹក​ស្ទឹង​នៅ​ក្បែរ​ព្រំដែន​​រវាង​ក្រិក​ និង​ម៉ាសេដ្វាន់",
        "ឥលុវណាឬកវីមានកម្រិតខ្ពស់"  # Test with independent vowels
    ]
    
    methods = [
        ('Basic regex', khmer_syllables),
        ('Advanced regex', khmer_syllables_advanced),
        ('Non-regex original', khmer_syllables_no_regex),
        ('Non-regex fast', khmer_syllables_no_regex_fast),
    ]
    
    print("🧪 TESTING FIXED KHMER SEGMENTATION FUNCTIONS")
    print("=" * 60)
    
    all_consistent = True
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n📝 TEST CASE {i}: {test_text[:50]}...")
        print("-" * 40)
        
        results = []
        for method_name, method_func in methods:
            try:
                result = method_func(test_text)
                results.append((method_name, result, len(result)))
                print(f"{method_name:20}: {len(result):3d} syllables - {result[:5]}...")
            except Exception as e:
                print(f"{method_name:20}: ERROR - {e}")
                all_consistent = False
        
        # Check if all results are the same
        if len(set(len(r[1]) for r in results)) == 1:
            if len(set(str(r[1]) for r in results)) == 1:
                print("✅ All methods consistent!")
            else:
                print("❌ Same count, different results!")
                all_consistent = False
        else:
            print("❌ Different syllable counts!")
            all_consistent = False
    
    print("\n" + "=" * 60)
    if all_consistent:
        print("🎉 ALL TESTS PASSED! All methods are now consistent.")
    else:
        print("❌ Some inconsistencies remain.")
    
    return all_consistent

if __name__ == "__main__":
    test_problematic_texts() 