import regex
import unicodedata

# Fast non-regex helper functions for Unicode character checking
def is_khmer_consonant(char):
    """Check if character is a Khmer consonant (U+1780-U+17A2)"""
    return 0x1780 <= ord(char) <= 0x17A2

def is_khmer_subscript(char):
    """Check if character is Khmer coeng/subscript (U+17D2)"""
    return ord(char) == 0x17D2

def is_khmer_vowel(char):
    """Check if character is a Khmer vowel (independent or dependent)"""
    code = ord(char)
    return (0x17A5 <= code <= 0x17B3) or (0x17B6 <= code <= 0x17C8)

def is_khmer_independent_vowel(char):
    """Check if character is a Khmer independent vowel (U+17A5-U+17B3)"""
    return 0x17A5 <= ord(char) <= 0x17B3

def is_khmer_dependent_vowel(char):
    """Check if character is a Khmer dependent vowel (U+17B6-U+17C8)"""
    return 0x17B6 <= ord(char) <= 0x17C8

def is_khmer_diacritic(char):
    """Check if character is a Khmer diacritic (U+17C9-U+17D1, U+17DD)"""
    code = ord(char)
    return (0x17C9 <= code <= 0x17D1) or code == 0x17DD

def is_khmer_digit(char):
    """Check if character is a Khmer digit (U+17E0-U+17E9)"""
    return 0x17E0 <= ord(char) <= 0x17E9

def is_khmer_symbol(char):
    """Check if character is a Khmer symbol (U+17D4-U+17DC)"""
    return 0x17D4 <= ord(char) <= 0x17DC

def is_khmer_character(char):
    """Check if character is any Khmer character"""
    return 0x1780 <= ord(char) <= 0x17FF

def khmer_syllables_no_regex(text):
    """
    Khmer syllable segmentation without regex.
    Uses direct Unicode character checking for better performance on large text.
    
    Args:
        text (str): Khmer text to segment
        
    Returns:
        list: List of Khmer syllables
        
    Example:
        >>> khmer_syllables_no_regex("អ្នកគ្រួបង្រៀនភាសាខ្មែរ")
        ['អ្ន', 'ក', 'គ្រួ', 'ប', 'ង្រៀ', 'ន', 'ភា', 'សា', 'ខ្មែ', 'រ']
    """
    if not text:
        return []
    
    # Remove zero-width spaces for consistent results across all methods
    text = text.replace('\u200b', '')
    
    syllables = []
    i = 0
    length = len(text)
    
    while i < length:
        current_syllable = ""
        
        # Handle whitespace as single tokens
        if text[i].isspace():
            start_space = i
            while i < length and text[i].isspace():
                i += 1
            syllables.append(text[start_space:i])
            continue
        
        # Handle Khmer characters
        if is_khmer_character(text[i]):
            
            # Case 1: Independent vowel - check if followed by coeng + consonant
            if is_khmer_independent_vowel(text[i]):
                current_syllable += text[i]
                i += 1
                
                # Check if followed by coeng + consonant - if so, treat as one syllable
                if i < length and is_khmer_subscript(text[i]):
                    # Include the coeng
                    current_syllable += text[i]
                    i += 1
                    # Include the following consonant if present
                    if i < length and is_khmer_consonant(text[i]):
                        current_syllable += text[i]
                        i += 1
                        
                        # Continue with any vowels/diacritics after the consonant
                        while i < length and (is_khmer_dependent_vowel(text[i]) or is_khmer_diacritic(text[i])):
                            current_syllable += text[i]
                            i += 1
                # If not followed by coeng, just continue with regular vowel/diacritic handling
                else:
                    while i < length and (is_khmer_dependent_vowel(text[i]) or is_khmer_diacritic(text[i])):
                        current_syllable += text[i]
                        i += 1
            
            # Case 2: Consonant-based syllable  
            elif is_khmer_consonant(text[i]):
                current_syllable += text[i]
                i += 1
                
                # Handle subscript consonants (coeng + consonant)
                while i < length and is_khmer_subscript(text[i]):
                    current_syllable += text[i]  # Add coeng
                    i += 1
                    if i < length and is_khmer_consonant(text[i]):
                        current_syllable += text[i]  # Add subscript consonant
                        i += 1
                
                # Handle dependent vowels and diacritics (but NOT independent vowels)
                while i < length and (is_khmer_dependent_vowel(text[i]) or is_khmer_diacritic(text[i])):
                    current_syllable += text[i]
                    i += 1
            
            # Case 3: Standalone dependent vowel or diacritic
            elif is_khmer_dependent_vowel(text[i]) or is_khmer_diacritic(text[i]):
                while i < length and (is_khmer_dependent_vowel(text[i]) or is_khmer_diacritic(text[i])):
                    current_syllable += text[i]
                    i += 1
            
            # Case 4: Khmer digits (consecutive digits as single token)
            elif is_khmer_digit(text[i]):
                while i < length and is_khmer_digit(text[i]):
                    current_syllable += text[i]
                    i += 1
            
            # Case 5: Khmer symbols (each symbol as separate token)
            elif is_khmer_symbol(text[i]):
                current_syllable += text[i]
                i += 1
            
            # Case 6: Other Khmer characters
            else:
                current_syllable += text[i]
                i += 1
        
        # Handle non-Khmer characters
        else:
            # Collect consecutive non-Khmer characters to match regex behavior
            start_pos = i
            while i < length and not is_khmer_character(text[i]) and not text[i].isspace():
                i += 1
            current_syllable = text[start_pos:i]
        
        if current_syllable:
            syllables.append(current_syllable)
    
    return syllables

def segment_paragraph_to_subwords_no_regex(text, separator="_"):
    """
    Fast version of paragraph segmentation using non-regex implementation.
    
    Args:
        text (str): Input Khmer text or paragraph
        separator (str): Character to use for joining syllables (default: "_")
        
    Returns:
        str: Segmented text with syllables separated by the specified separator
    """
    syllables = khmer_syllables_no_regex(text)
    return separator.join(syllables)

def segment_paragraph_to_subwords_no_regex_fast(text, separator="_"):
    """
    Fastest version of paragraph segmentation using optimized non-regex implementation.
    This is an alias for segment_paragraph_to_subwords_no_regex for compatibility.
    
    Args:
        text (str): Input Khmer text or paragraph
        separator (str): Character to use for joining syllables (default: "_")
        
    Returns:
        str: Segmented text with syllables separated by the specified separator
    """
    syllables = khmer_syllables_no_regex_fast(text)
    return separator.join(syllables)

def segment_paragraph_to_subwords_optimized(text, separator="_"):
    """
    Optimized version of paragraph segmentation (alias for the fast non-regex version).
    
    Args:
        text (str): Input Khmer text or paragraph
        separator (str): Character to use for joining syllables (default: "_")
        
    Returns:
        str: Segmented text with syllables separated by the specified separator
    """
    syllables = khmer_syllables_no_regex_fast(text)
    return separator.join(syllables)

def khmer_syllables(text):
    """
    Segment Khmer text into syllables using proper Khmer script patterns.
    
    Khmer syllables typically follow this structure:
    - Initial consonant (with possible subscript consonants)
    - Optional vowels and diacritics
    - Optional final consonant
    
    Args:
        text (str): Khmer text to segment
        
    Returns:
        list: List of Khmer syllables
        
    Example:
        >>> khmer_syllables("អ្នកគ្រួបង្រៀនភាសាខ្មែរ")
        ['អ្ន', 'ក', 'គ្រួ', 'ប', 'ង្រៀ', 'ន', 'ភា', 'សា', 'ខ្មែ', 'រ']
    """
    
    # Remove zero-width spaces for consistent results across all methods
    if not text:
        return []
    text = text.replace('\u200b', '')
    
    # Enhanced regex pattern that handles subscript consonants correctly
    pattern = (
        r'(?:'
        # Whitespace as single tokens
        r'\s+'
        r'|'
        # Independent vowels with optional coeng + consonant combinations
        r'[\u17A5-\u17B3](?:\u17D2[\u1780-\u17A2][\u17B6-\u17D1]*)?'
        r'|'
        # Main consonant with subscript consonants and vowels - EXPANDED to handle cases like ក្ុ
        r'[\u1780-\u17A2](?:\u17D2[\u1780-\u17A2])*(?:[\u17B6-\u17D1]|\u17D2[\u17B6-\u17D1])*'
        r'|'
        # Standalone dependent vowels/diacritics
        r'[\u17B6-\u17C8\u17C9-\u17D1\u17DD]+'
        r'|'
        # Standalone subscript marker (coeng) - comes after main patterns
        r'\u17D2'
        r'|'
        # Digits (consecutive digits as single token)
        r'[\u17E0-\u17E9]+'
        r'|'
        # Symbols (each symbol as separate token)
        r'[\u17D4-\u17DC]'
        r'|'
        # Consecutive non-Khmer characters as single units
        r'[^\u1780-\u17FF\s]+'
        r')'
    )
    
    # Use single regex.findall to get all tokens including spaces
    matches = regex.findall(pattern, text)
    
    # Filter out empty matches and return
    result = [match for match in matches if match]
    return result

def khmer_syllables_with_underscores(text):
    """
    Segment Khmer text into syllables and join with underscores.
    
    Args:
        text (str): Khmer text to segment
        
    Returns:
        str: Syllables joined with underscores
        
    Example:
        >>> khmer_syllables_with_underscores("អ្នកគ្រួបង្រៀនភាសាខ្មែរ")
        'អ្ន_ក_គ្រួ_ប_ង្រៀ_ន_ភា_សា_ខ្មែ_រ'
    """
    syllables = khmer_syllables(text)
    return '_'.join(syllables)

def khmer_syllables_advanced(text):
    """
    More advanced Khmer syllable segmentation using a different approach.
    This tries to better handle complex Khmer syllable structures.
    
    Args:
        text (str): Khmer text to segment
        
    Returns:
        list: List of Khmer syllables
    """
    
    # Remove zero-width spaces for consistent results across all methods
    if not text:
        return []
    text = text.replace('\u200b', '')
    
    # Enhanced regex pattern that handles subscript consonants correctly
    pattern = (
        r'(?:'
        # Whitespace as single tokens
        r'\s+'
        r'|'
        # Independent vowel with optional coeng + consonant combination
        r'[\u17A5-\u17B3](?:\u17D2[\u1780-\u17A2][\u17B6-\u17D1]*)?'
        r'|'
        # Khmer syllable: consonant + optional subscripts + optional vowels/diacritics - EXPANDED
        r'[\u1780-\u17A2]'              # Base consonant
        r'(?:\u17D2[\u1780-\u17A2])*'   # Optional subscript consonants
        r'(?:[\u17B6-\u17D1]|\u17D2[\u17B6-\u17D1])*'  # Optional dependent vowels and diacritics, including coeng+vowel
        r'|'
        # Standalone dependent vowels/diacritics
        r'[\u17B6-\u17C8\u17C9-\u17D1\u17DD]+'
        r'|'
        # Standalone subscript marker (coeng) - comes after main patterns
        r'\u17D2'
        r'|'
        # Digits (consecutive digits as single token)
        r'[\u17E0-\u17E9]+'
        r'|'
        # Symbols (each symbol as separate token)
        r'[\u17D4-\u17DC]'
        r'|'
        # Consecutive non-Khmer characters as single units
        r'[^\u1780-\u17FF\s]+'
        r')'
    )
    
    # Use single regex.findall to get all tokens including spaces
    matches = regex.findall(pattern, text)
    
    # Filter out empty matches and return
    result = [match for match in matches if match]
    return result

def segment_paragraph_to_subwords(text, method="basic", separator="_"):
    """
    Main function to segment a Khmer paragraph into sub-words/syllables.
    
    Args:
        text (str): Input Khmer text or paragraph
        method (str): Segmentation method - "basic" or "advanced"
        separator (str): Character to use for joining syllables (default: "_")
        
    Returns:
        str: Segmented text with syllables separated by the specified separator
        
    Example:
        >>> segment_paragraph_to_subwords("អ្នកគ្រួបង្រៀនភាសាខ្មែរ")
        'អ្ន_ក_គ្រួ_ប_ង្រៀ_ន_ភា_សា_ខ្មែ_រ'
        
        >>> segment_paragraph_to_subwords("អ្នកគ្រួបង្រៀនភាសាខ្មែរ", separator="|")
        'អ្ន|ក|គ្រួ|ប|ង្រៀ|ន|ភា|សា|ខ្មែ|រ'
    """
    if method == "advanced":
        syllables = khmer_syllables_advanced(text)
    else:
        syllables = khmer_syllables(text)
    
    return separator.join(syllables)

def khmer_syllables_no_regex_fast(text):
    """
    Non-regex Khmer syllable segmentation.
    Minimizes function calls and uses faster character checking.
    
    Args:
        text (str): Khmer text to segment
        
    Returns:
        list: List of Khmer syllables
    """
    if not text:
        return []
    
    # Remove zero-width spaces for consistent results across all methods
    text = text.replace('\u200b', '')
    
    syllables = []
    i = 0
    length = len(text)
    
    while i < length:
        char_code = ord(text[i])
        
        # Handle whitespace as single tokens
        if text[i].isspace():
            start_space = i
            while i < length and text[i].isspace():
                i += 1
            syllables.append(text[start_space:i])
            continue
        
        start_pos = i
        
        # Handle Khmer characters (U+1780-U+17FF)
        if 0x1780 <= char_code <= 0x17FF:
            
            # Independent vowel - check if followed by coeng + consonant (FIXED: correct range U+17A5-U+17B3)
            if 0x17A5 <= char_code <= 0x17B3:
                i += 1
                # Check if followed by coeng + consonant
                if i < length and ord(text[i]) == 0x17D2:  # Coeng
                    i += 1
                    if i < length and 0x1780 <= ord(text[i]) <= 0x17A2:  # Consonant
                        i += 1
                        # Continue with any dependent vowels/diacritics after the consonant
                        while i < length:
                            char_code = ord(text[i])
                            if (0x17B6 <= char_code <= 0x17C8) or \
                               (0x17C9 <= char_code <= 0x17D1) or \
                               char_code == 0x17DD:
                                i += 1
                            else:
                                break
                # If not followed by coeng, handle any dependent vowels/diacritics
                else:
                    while i < length:
                        char_code = ord(text[i])
                        if (0x17B6 <= char_code <= 0x17C8) or \
                           (0x17C9 <= char_code <= 0x17D1) or \
                           char_code == 0x17DD:
                            i += 1
                        else:
                            break
            
            # Consonant-based syllable (U+1780-U+17A2)
            elif 0x1780 <= char_code <= 0x17A2:
                i += 1
                
                # Handle subscript consonants (coeng + consonant)
                while i < length:
                    if ord(text[i]) == 0x17D2:  # Coeng
                        i += 1
                        if i < length and 0x1780 <= ord(text[i]) <= 0x17A2:
                            i += 1
                        else:
                            break
                    else:
                        break
                
                # Handle dependent vowels and diacritics (but NOT independent vowels)
                while i < length:
                    char_code = ord(text[i])
                    # Only include dependent vowels (U+17B6-U+17C8), not independent vowels (U+17A5-U+17B3)
                    if (0x17B6 <= char_code <= 0x17C8) or \
                       (0x17C9 <= char_code <= 0x17D1) or \
                       char_code == 0x17DD:
                        i += 1
                    else:
                        break
            
            # Standalone dependent vowel/diacritic (not independent vowels)
            elif (0x17B6 <= char_code <= 0x17C8) or \
                 (0x17C9 <= char_code <= 0x17D1) or \
                 char_code == 0x17DD:
                i += 1
                # Continue collecting vowels/diacritics
                while i < length:
                    char_code = ord(text[i])
                    if (0x17B6 <= char_code <= 0x17C8) or \
                       (0x17C9 <= char_code <= 0x17D1) or \
                       char_code == 0x17DD:
                        i += 1
                    else:
                        break
            
            # Digits (consecutive digits as single token)
            elif 0x17E0 <= char_code <= 0x17E9:
                i += 1
                # Continue collecting consecutive digits
                while i < length:
                    char_code = ord(text[i])
                    if 0x17E0 <= char_code <= 0x17E9:
                        i += 1
                    else:
                        break
            
            # Symbols (each symbol as separate token)
            elif 0x17D4 <= char_code <= 0x17DC:
                i += 1
            
            # Other Khmer characters
            else:
                i += 1
        
        # Handle non-Khmer characters
        else:
            # Collect consecutive non-Khmer characters (excluding spaces) to match regex behavior
            while i < length and not (0x1780 <= ord(text[i]) <= 0x17FF) and not text[i].isspace():
                i += 1
        
        # Add the syllable if it's not empty
        syllable = text[start_pos:i]
        if syllable:
            syllables.append(syllable)
    
    return syllables

def main():
    # Test with the sample text
    sample = "អ្នកគ្រួបង្រៀនភាសាខ្មែរ"
    
    print("=== Khmer Syllable Segmentation Demo ===")
    print(f"Original text: {sample}")
    print()
    
    # Basic usage
    print("Using the main function:")
    result = segment_paragraph_to_subwords(sample)
    print(f"Result: {result}")
    print()
    
    # With different separator
    print("With different separator:")
    result_pipe = segment_paragraph_to_subwords(sample, separator="|")
    print(f"Result: {result_pipe}")
    print()
    
    print("Method 1 - Basic syllable segmentation:")
    syllables1 = khmer_syllables(sample)
    print("Syllables:", syllables1)
    print("With underscores:", khmer_syllables_with_underscores(sample))
    print()
    
    print("Method 2 - Advanced syllable segmentation:")
    syllables2 = khmer_syllables_advanced(sample)
    print("Syllables:", syllables2)
    print("With underscores:", '_'.join(syllables2))
    print()
    
    # Test with a longer paragraph
    paragraph = "ខ្ញុំចង់រៀនភាសាខ្មែរ។ តើអ្នកជួយខ្ញុំបានទេ? អរគុណច្រើន!"
    print("=== Paragraph Test ===")
    print("Original:", paragraph)
    print("Segmented:", segment_paragraph_to_subwords(paragraph))

if __name__ == "__main__":
    paragraph = "ខ្ញុំចង់រៀនភាសាខ្មែរឱ្យបានល្អ ឲ្យបានល្អ។ ក្រមុំលក់នំនៅភ្នំពេញ។"
    print("=== Paragraph Test ===")
    print("Original:", paragraph)
    print("Segmented:", segment_paragraph_to_subwords(paragraph, separator="|"))
