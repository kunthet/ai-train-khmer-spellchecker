# -*- coding: utf-8 -*-
"""
Example usage of Khmer syllable segmentation functions.
This demonstrates how to split Khmer text into sub-words/syllables.
"""

from subword_cluster import segment_paragraph_to_subwords, khmer_syllables, khmer_syllables_with_underscores

def main():
    print("Khmer Syllable Segmentation Examples")
    print("=" * 50)
    
    # Example 1: Basic usage
    text1 = "អ្នកគ្រួបង្រៀនភាសាខ្មែរ"
    print(f"Text: {text1}")
    result1 = segment_paragraph_to_subwords(text1)
    print(f"Segmented: {result1}")
    print()
    
    # Example 2: With different separator
    print("With different separators:")
    print(f"With pipes: {segment_paragraph_to_subwords(text1, separator='|')}")
    print(f"With spaces: {segment_paragraph_to_subwords(text1, separator=' ')}")
    print(f"With hyphens: {segment_paragraph_to_subwords(text1, separator='-')}")
    print()
    
    # Example 3: Paragraph
    paragraph = "ខ្ញុំចង់រៀនភាសាខ្មែរ។ តើអ្នកជួយខ្ញុំបានទេ?"
    print(f"Paragraph: {paragraph}")
    print(f"Segmented: {segment_paragraph_to_subwords(paragraph)}")
    print()
    
    # Example 4: Using individual functions
    print("Using individual functions:")
    syllables = khmer_syllables(text1)
    print(f"Syllables list: {syllables}")
    print(f"Count: {len(syllables)} syllables")
    print()
    
    # Example 5: Show expected vs actual from original problem
    expected = "អ្ន_ក_គ្រួ_ប_ង្រៀ_ន_ភា_សា_ខ្មែ_រ"
    actual = segment_paragraph_to_subwords(text1)
    print("Comparison with expected output:")
    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    print(f"Match: {expected == actual}")

if __name__ == "__main__":
    main() 