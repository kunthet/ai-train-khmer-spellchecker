# -*- coding: utf-8 -*-
"""
Simple test for user's specific requirements
"""

from subword_cluster import khmer_syllables, khmer_syllables_no_regex_fast

def main():
    print("Testing user's specific requirements:")
    print("ឲ්យ -> ['ឲ්យ']")
    print("ឱ්យ -> ['ឱ្យ']")
    print()
    
    # Test ឲ្យ
    test1_regex = khmer_syllables('ឲ្យ')
    test1_nonregex = khmer_syllables_no_regex_fast('ឲ្យ')
    
    print("Test 1: ឲ្យ")
    print(f"Regex result:     {test1_regex}")
    print(f"Non-regex result: {test1_nonregex}")
    print(f"Expected:         ['ឲ្យ']")
    print(f"Regex passes:     {test1_regex == ['ឲ្យ']}")
    print(f"Non-regex passes: {test1_nonregex == ['ឲ្យ']}")
    print()
    
    # Test ឱ្យ  
    test2_regex = khmer_syllables('ឱ្យ')
    test2_nonregex = khmer_syllables_no_regex_fast('ឱ្យ')
    
    print("Test 2: ឱ្យ")
    print(f"Regex result:     {test2_regex}")
    print(f"Non-regex result: {test2_nonregex}")
    print(f"Expected:         ['ឱ្យ']")
    print(f"Regex passes:     {test2_regex == ['ឱ្យ']}")
    print(f"Non-regex passes: {test2_nonregex == ['ឱ្យ']}")
    print()
    
    # Overall result
    all_pass = (test1_regex == ['ឲ្យ'] and test1_nonregex == ['ឲ្យ'] and
                test2_regex == ['ឱ្យ'] and test2_nonregex == ['ឱ្យ'])
    
    print("=" * 50)
    if all_pass:
        print("SUCCESS: Both requirements are met!")
        print("✅ ឲ្យ correctly produces ['ឲ្យ']")
        print("✅ ឱ្យ correctly produces ['ឱ្យ']")
    else:
        print("FAILURE: Requirements not met")
        
    print("=" * 50)
    
    return all_pass

if __name__ == "__main__":
    main() 