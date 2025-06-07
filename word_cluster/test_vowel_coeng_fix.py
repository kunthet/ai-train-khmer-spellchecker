# -*- coding: utf-8 -*-
"""
Test script to verify the fix for independent vowel + coeng + consonant combinations
"""

from subword_cluster import (
    khmer_syllables,
    khmer_syllables_no_regex_fast,
    khmer_syllables_advanced
)

def test_vowel_coeng_combinations():
    """Test independent vowel + coeng + consonant combinations"""
    
    test_cases = [
        # The main cases the user reported
        ("ឲ្យ", "Independent vowel QOO TYPE TWO + coeng + YO"),
        ("ឱ្យ", "Independent vowel QOO TYPE ONE + coeng + YO"),
        
        # Additional test cases for completeness
        ("ឯ្យ", "Independent vowel QE + coeng + YO"),
        ("ឰ្យ", "Independent vowel QAI + coeng + YO"),
        ("ឲ្រ", "Independent vowel QOO TYPE TWO + coeng + RO"),
        ("ឱ្រ", "Independent vowel QOO TYPE ONE + coeng + RO"),
        
        # In context with other text
        ("Apple ឲ្យប្រើ", "Mixed text with ឲ្យ"),
        ("ខ្ញុំឲ្យគេ", "Khmer text with ឲ្យ"),
        ("ឱ្យនេះ", "ឱ្យ at beginning"),
    ]
    
    print("=" * 80)
    print("TESTING INDEPENDENT VOWEL + COENG + CONSONANT COMBINATIONS")
    print("=" * 80)
    print()
    
    all_passed = True
    
    for text, description in test_cases:
        print(f"Test: {description}")
        print(f"Input: {text} (chars: {[char for char in text]})")
        
        # Test all implementations
        regex_result = khmer_syllables(text)
        non_regex_result = khmer_syllables_no_regex_fast(text)
        advanced_result = khmer_syllables_advanced(text)
        
        print(f"Regex result:     {regex_result}")
        print(f"Non-regex result: {non_regex_result}")
        print(f"Advanced result:  {advanced_result}")
        
        # Check if the key vowel+coeng combinations are preserved
        for syllable in text.split():
            if any(syllable.startswith(vowel) and '្' in syllable for vowel in ['ឲ', 'ឱ', 'ឯ', 'ឰ']):
                expected_intact = syllable
                
                # Check if any implementation keeps it intact
                regex_has_intact = expected_intact in regex_result
                non_regex_has_intact = expected_intact in non_regex_result
                advanced_has_intact = expected_intact in advanced_result
                
                print(f"  Expected intact: '{expected_intact}'")
                print(f"  Regex preserves:     {'✅' if regex_has_intact else '❌'}")
                print(f"  Non-regex preserves: {'✅' if non_regex_has_intact else '❌'}")
                print(f"  Advanced preserves:  {'✅' if advanced_has_intact else '❌'}")
                
                if not (regex_has_intact and non_regex_has_intact):
                    all_passed = False
        
        print()
    
    print("=" * 80)
    print(f"OVERALL RESULT: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("=" * 80)
    
    return all_passed

def test_specific_cases():
    """Test the exact cases user mentioned"""
    
    print("\n" + "=" * 50)
    print("USER'S SPECIFIC REQUIREMENTS:")
    print("ឲ្យ -> ['ឲ្យ']")
    print("ឱ្យ -> ['ឱ្យ']")
    print("=" * 50)
    
    # Test ឲ្យ
    result1_regex = khmer_syllables('ឲ្យ')
    result1_nonregex = khmer_syllables_no_regex_fast('ឲ្យ')
    
    print(f"ឲ្យ regex result:     {result1_regex}")
    print(f"ឲ្យ non-regex result: {result1_nonregex}")
    print(f"ឲ្យ passes: {'✅' if result1_regex == ['ឲ្យ'] and result1_nonregex == ['ឲ្យ'] else '❌'}")
    print()
    
    # Test ឱ្យ
    result2_regex = khmer_syllables('ឱ្យ')
    result2_nonregex = khmer_syllables_no_regex_fast('ឱ្យ')
    
    print(f"ឱ្យ regex result:     {result2_regex}")
    print(f"ឱ្យ non-regex result: {result2_nonregex}")
    print(f"ឱ្យ passes: {'✅' if result2_regex == ['ឱ្យ'] and result2_nonregex == ['ឱ្យ'] else '❌'}")
    
    # Check both
    both_pass = (result1_regex == ['ឲ្យ'] and result1_nonregex == ['ឲ្យ'] and 
                 result2_regex == ['ឱ្យ'] and result2_nonregex == ['ឱ្យ'])
    
    print(f"\n{'🎉 SUCCESS: Both cases work correctly!' if both_pass else '⚠️  FAILURE: Issues remain'}")
    
    return both_pass

if __name__ == "__main__":
    # Run the comprehensive test
    comprehensive_passed = test_vowel_coeng_combinations()
    
    # Run the specific user requirements test
    specific_passed = test_specific_cases()
    
    print(f"\n🎯 FINAL RESULT:")
    print(f"Comprehensive tests: {'PASSED' if comprehensive_passed else 'FAILED'}")
    print(f"User requirements:   {'PASSED' if specific_passed else 'FAILED'}") 