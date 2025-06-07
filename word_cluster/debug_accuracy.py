# Debug script to find accuracy issues
from subword_cluster import khmer_syllables, khmer_syllables_no_regex

def debug_accuracy():
    # Test with spaces and punctuation
    test = "ខ្ញុំចង់រៀនភាសាខ្មែរ។ ក្រមុំលក់នំនៅភ្នំពេញ។ តើអ្នកជួយខ្ញុំបានទេ? អរគុណច្រើន!"
    
    print("=== Debugging Accuracy Issue ===")
    print(f"Test: {test}")
    print()
    
    regex_result = khmer_syllables(test)
    fast_result = khmer_syllables_no_regex(test)
    
    print(f"Regex count: {len(regex_result)}")
    print(f"Fast count:  {len(fast_result)}")
    print()
    
    print("Regex result:")
    for i, syllable in enumerate(regex_result):
        print(f"  {i:2d}: '{syllable}' (len={len(syllable)})")
    
    print("\nFast result:")
    for i, syllable in enumerate(fast_result):
        print(f"  {i:2d}: '{syllable}' (len={len(syllable)})")
    
    # Find differences
    print("\nDifferences:")
    max_len = max(len(regex_result), len(fast_result))
    differences = 0
    for i in range(max_len):
        r = regex_result[i] if i < len(regex_result) else "[MISSING]"
        f = fast_result[i] if i < len(fast_result) else "[MISSING]"
        if r != f:
            print(f"  {i:2d}: '{r}' vs '{f}'")
            differences += 1
    
    if differences == 0:
        print("  No differences found!")

if __name__ == "__main__":
    debug_accuracy() 