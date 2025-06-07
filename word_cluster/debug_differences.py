# -*- coding: utf-8 -*-
"""
Debug script to analyze specific differences between regex and non-regex implementations
"""

from subword_cluster import (
    khmer_syllables,
    khmer_syllables_no_regex_fast,
    khmer_syllables_advanced
)

def debug_specific_case(text, case_name):
    """Debug a specific text case"""
    print(f"\n=== {case_name} ===")
    print(f"Input: {repr(text)}")
    print(f"Visible: {text}")
    print()
    
    # Test all implementations
    regex_result = khmer_syllables(text)
    non_regex_result = khmer_syllables_no_regex_fast(text)
    advanced_result = khmer_syllables_advanced(text)
    
    print(f"Regex basic result      ({len(regex_result):3d}): {regex_result}")
    print(f"Non-regex result        ({len(non_regex_result):3d}): {non_regex_result}")
    print(f"Regex advanced result   ({len(advanced_result):3d}): {advanced_result}")
    
    # Find differences
    if regex_result != non_regex_result:
        print("\n🔍 DIFFERENCE FOUND between regex and non-regex!")
        
        # Show character-by-character analysis
        print(f"\nCharacter analysis:")
        for i, char in enumerate(text):
            print(f"  {i:2d}: {repr(char):8s} U+{ord(char):04X} {char}")
        
        # Compare results item by item
        max_len = max(len(regex_result), len(non_regex_result))
        print(f"\nItem-by-item comparison:")
        for i in range(max_len):
            regex_item = regex_result[i] if i < len(regex_result) else "<MISSING>"
            nonregex_item = non_regex_result[i] if i < len(non_regex_result) else "<MISSING>"
            
            status = "✅" if regex_item == nonregex_item else "❌"
            print(f"  {i:2d}: {status} Regex: {repr(regex_item):15s} | Non-regex: {repr(nonregex_item):15s}")
    else:
        print("✅ Results match!")
    
    print("=" * 80)

def main():
    # Test cases from the debug file
    test_cases = [
        ("Apple ឲ្យប្រើ", "English + Khmer"),
        ("Sabay Digital Corporation ជា", "Multi-word English + Khmer"),
        ("ជា\u200bក្រុម", "Zero-width joiner case"),
        ("១០. Taew គឺ", "Number + dot + English + Khmer"),
        ("៥. ដំបូង", "Khmer number + dot + Khmer"),
        ("Thon មិន", "English name + Khmer"),
        ("ឲ្យ", "Simple Khmer with subscript"),
        ("ហ៊ុន", "Khmer with diacritic"),
        ("A.អាកំបាំង", "Letter + dot + Khmer"),
        ("ម៉ាញ និង", "Khmer words with space"),
        ("ជួនកាល គេ", "Khmer words separated by space"),
        ("Reuters/Alexandros", "English with slash"),
        ("ឆ្លងឆ្លើយ", "Khmer with subscripts"),
        ("ខ្ញុំ", "Khmer with Nikahit"),
        ("ព្រហ្ម", "Khmer with multiple subscripts"),
        ("។", "Khmer punctuation"),
        ("១២៣", "Khmer digits"),
        ("English", "Pure English"),
        ("ាុំ", "Khmer vowels only"),
        ("្រ", "Khmer subscript + consonant"),
    ]
    
    print("🔍 KHMER SYLLABLE SEGMENTATION DEBUG ANALYSIS")
    print("=" * 80)
    
    for text, description in test_cases:
        debug_specific_case(text, description)

if __name__ == "__main__":
    main() 