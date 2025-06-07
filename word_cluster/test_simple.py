# Simple test to verify Khmer syllable segmentation works
from subword_cluster import segment_paragraph_to_subwords, khmer_syllables

def test_basic_functionality():
    # Test case from the original problem
    text = "អ្នកគ្រួបង្រៀនភាសាខ្មែរ"
    expected = "អ្ន_ក_គ្រួ_ប_ង្រៀ_ន_ភា_សា_ខ្មែ_រ"
    
    # Test the main function
    result = segment_paragraph_to_subwords(text)
    
    # Test syllable count
    syllables = khmer_syllables(text)
    
    print("Test Results:")
    print(f"✓ Function executes without error")
    print(f"✓ Expected 10 syllables, got: {len(syllables)}")
    print(f"✓ Result matches expected format: {result == expected}")
    print(f"✓ Can use different separators: {len(segment_paragraph_to_subwords(text, separator='|').split('|'))} parts")
    
    # Test with different separator
    pipe_result = segment_paragraph_to_subwords(text, separator="|")
    space_result = segment_paragraph_to_subwords(text, separator=" ")
    
    assert len(syllables) == 10, f"Expected 10 syllables, got {len(syllables)}"
    assert result == expected, "Result doesn't match expected output"
    assert len(pipe_result.split("|")) == 10, "Pipe separator test failed"
    assert len(space_result.split(" ")) == 10, "Space separator test failed"
    
    print("✓ All tests passed!")
    return True

if __name__ == "__main__":
    test_basic_functionality() 