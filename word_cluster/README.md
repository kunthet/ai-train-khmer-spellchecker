# Khmer Syllable Segmentation

This project provides tools for segmenting Khmer text into syllables/sub-words, which is essential for natural language processing tasks with Khmer text.

## Problem Solved

The original issue was that using simple grapheme cluster patterns (`\X`) incorrectly split Khmer syllables. For example:

- **Input:** `អ្នកគ្រួបង្រៀនភាសាខ្មែរ`
- **Incorrect output:** `['អ្', 'ន', 'ក', 'គ្', 'រួ', 'ប', 'ង្', 'រៀ', 'ន', ...]`
- **Correct output:** `['អ្ន', 'ក', 'គ្រួ', 'ប', 'ង្រៀ', 'ន', 'ភា', 'សា', 'ខ្មែ', 'រ']`

## Implementation Options

This project provides **three different implementations** with different performance and dependency characteristics:

### 1. Regex Version (Recommended) ⚡
```python
from subword_cluster import segment_paragraph_to_subwords

result = segment_paragraph_to_subwords("អ្នកគ្រួបង្រៀនភាសាខ្មែរ")
# Output: អ្ន_ក_គ្រួ_ប_ង្រៀ_ន_ភា_សា_ខ្មែ_រ
```
- **Performance:** Fastest (baseline)
- **Dependencies:** Requires `regex` library
- **Use for:** Production use, best performance

### 2. Optimized Non-Regex Version 🔧
```python
from subword_cluster import segment_paragraph_to_subwords_optimized

result = segment_paragraph_to_subwords_optimized("អ្នកគ្រួបង្រៀនភាសាខ្មែរ")
# Output: អ្ន_ក_គ្រួ_ប_ង្រៀ_ន_ភា_សា_ខ្មែ_រ
```
- **Performance:** ~1.7x slower than regex
- **Dependencies:** No external dependencies
- **Use for:** When regex dependency must be avoided

### 3. Fast Non-Regex Version 📚
```python
from subword_cluster import segment_paragraph_to_subwords_fast

result = segment_paragraph_to_subwords_fast("អ្នកគ្រួបង្រៀនភាសាខ្មែរ")
# Output: អ្ន_ក_គ្រួ_ប_ង្រៀ_ន_ភា_សា_ខ្មែ_រ
```
- **Performance:** ~2.7x slower than regex
- **Dependencies:** No external dependencies
- **Use for:** Educational purposes, understanding the algorithm

## Performance Comparison

Based on testing with 77,000 characters:

| Implementation | Time (seconds) | Relative Speed |
|---------------|----------------|----------------|
| Regex         | 0.034         | 1.00x (fastest) |
| Optimized     | 0.060         | 0.57x          |
| Fast          | 0.091         | 0.37x          |

## Quick Usage

```python
from subword_cluster import segment_paragraph_to_subwords

# Basic usage - segment and join with underscores
text = "ខ្ញុំចង់រៀនភាសាខ្មែរ។ ក្រមុំលក់នំនៅភ្នំពេញ។"
result = segment_paragraph_to_subwords(text)
print(result)
# Output: ខ្ញុំ_ច_ង់_រៀ_ន_ភា_សា_ខ្មែ_រ_។_ក្រ_មុំ_ល_ក់_នំ_នៅ_ភ្នំ_ពេ_ញ_។

# Use different separator
result = segment_paragraph_to_subwords(text, separator="|")
print(result)
# Output: ខ្ញុំ|ច|ង់|រៀ|ន|ភា|សា|ខ្មែ|រ|។|ក្រ|មុំ|ល|ក់|នំ|នៅ|ភ្នំ|ពេ|ញ|។

# Get syllables as list
from subword_cluster import khmer_syllables
syllables = khmer_syllables(text)
print(syllables)
# Output: ['ខ្ញុំ', 'ច', 'ង់', 'រៀ', 'ន', 'ភា', 'សា', 'ខ្មែ', 'រ', '។', 'ក្រ', 'មុំ', 'ល', 'ក់', 'នំ', 'នៅ', 'ភ្នំ', 'ពេ', 'ញ', '។']
```

## Advanced Usage

```python
# For maximum performance (recommended)
from subword_cluster import khmer_syllables, segment_paragraph_to_subwords

# For no-dependency environments
from subword_cluster import khmer_syllables_optimized, segment_paragraph_to_subwords_optimized

# For educational/research purposes
from subword_cluster import khmer_syllables_fast, segment_paragraph_to_subwords_fast

# All functions have the same API
text = "អ្នកគ្រួបង្រៀនភាសាខ្មែរ"
syllables = khmer_syllables(text)               # Fastest
syllables = khmer_syllables_optimized(text)     # No dependencies
syllables = khmer_syllables_fast(text)          # Educational
```

## Features

✅ **Accurate Khmer Script Handling**: Properly handles subscripts, vowels, and diacritics  
✅ **Multiple Implementation Options**: Choose based on your performance/dependency needs  
✅ **Nikahit Support**: Correctly handles ំ (U+17C6) character  
✅ **Configurable Output**: Different separators and output formats  
✅ **Well Tested**: All implementations produce identical results  
✅ **Production Ready**: Used for NLP tasks and spell checking  

## Installation

```bash
# For regex version (recommended)
pip install regex

# No additional dependencies needed for non-regex versions
```

## Files

- `subword_cluster.py` - Main implementation with all three versions
- `performance_test.py` - Performance comparison tool
- `example_usage.py` - Comprehensive usage examples
- `test_simple.py` - Basic functionality tests

## Contributing

The implementation follows Khmer Unicode standards (U+1780-U+17FF) and handles:
- Consonants (U+1780-U+17A2)
- Subscripts/Coeng (U+17D2)
- Vowels (U+17B6-U+17C8) including Nikahit (U+17C6)
- Diacritics (U+17C9-U+17D1, U+17DD)
- Digits and symbols

## License

Open source - feel free to use and contribute! 