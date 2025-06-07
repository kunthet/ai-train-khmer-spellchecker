# -*- coding: utf-8 -*-
"""
Debug the specific issue with រឱ clustering
"""

from subword_cluster import khmer_syllables, khmer_syllables_no_regex_fast

def main():
    # The problematic text
    test_text = "ខ្មែរឱ្យ"
    
    print("Debug: រឱ clustering issue")
    print(f"Input: {test_text}")
    print(f"Expected: ['ខ្មែ', 'រ', 'ឱ្យ']")
    print()
    
    # Test both implementations
    regex_result = khmer_syllables(test_text)
    non_regex_result = khmer_syllables_no_regex_fast(test_text)
    
    print(f"Regex result:     {regex_result}")
    print(f"Non-regex result: {non_regex_result}")
    print()
    
    # Character analysis
    print("Character breakdown:")
    for i, char in enumerate(test_text):
        print(f"  {i}: '{char}' U+{ord(char):04X}")
    
    print()
    print("Issue: 'រឱ' should be split as 'រ' + 'ឱ្យ'")
    
    # Test simpler cases
    print("\nSimpler test cases:")
    simple_tests = ["រ", "ឱ្យ", "រឱ", "រឱ្យ"]
    
    for test in simple_tests:
        regex_res = khmer_syllables(test)
        nonregex_res = khmer_syllables_no_regex_fast(test)
        print(f"{test:4} -> Regex: {regex_res}, Non-regex: {nonregex_res}")

if __name__ == "__main__":
    main() 