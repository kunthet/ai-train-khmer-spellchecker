# -*- coding: utf-8 -*-
"""
Test the final fix for the clustering issue
"""

from subword_cluster import segment_paragraph_to_subwords

def main():
    # The original problematic text
    text = "ខ្ញុំចង់រៀនភាសាខ្មែរឱ្យបានល្អ ឲ្យបានល្អ។ ក្រមុំលក់នំនៅភ្នំពេញ។"
    
    print("Testing the original paragraph with fixes:")
    print(f"Input: {text}")
    print()
    
    result = segment_paragraph_to_subwords(text, separator='|')
    print(f"Result: {result}")
    print()
    
    # Check if រឱ issue is fixed
    if "រឱ" in result:
        print("❌ ISSUE STILL EXISTS: Found 'រឱ' cluster")
    else:
        print("✅ ISSUE FIXED: No 'រឱ' cluster found")
    
    # Check if our target syllables are preserved
    if "ឲ្យ" in result and "ឱ្យ" in result:
        print("✅ SYLLABLES PRESERVED: Both 'ឲ្យ' and 'ឱ្យ' found intact")
    else:
        print("❌ SYLLABLES BROKEN: Missing 'ឲ្យ' or 'ឱ្យ'")
    
    print()
    print("Expected clusters that should be preserved:")
    print("- ខ្ញុំ (not ខ្ញុ)")
    print("- ខ្មែ (not ខ្ម + ែ)")
    print("- រ (standalone)")
    print("- ឱ្យ (not រឱ + យ)")
    print("- ឲ្យ (preserved)")
    print("- ភ្នំ (not ភ្ន + ំ)")

if __name__ == "__main__":
    main() 