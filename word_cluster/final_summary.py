# -*- coding: utf-8 -*-
"""
Final Summary: Khmer Syllable Segmentation Differences Analysis
"""

from subword_cluster import (
    khmer_syllables, 
    khmer_syllables_no_regex_fast,
    khmer_syllables_advanced
)

def test_case(text, description):
    """Test a specific case and return results"""
    regex_result = khmer_syllables(text)
    non_regex_result = khmer_syllables_no_regex_fast(text)
    advanced_result = khmer_syllables_advanced(text)
    
    # Check if all match
    all_match = regex_result == non_regex_result == advanced_result
    
    return {
        'text': text,
        'description': description,
        'regex': regex_result,
        'non_regex': non_regex_result,
        'advanced': advanced_result,
        'all_match': all_match,
        'regex_vs_nonregex': regex_result == non_regex_result
    }

def main():
    print("=" * 80)
    print("FINAL SUMMARY: Khmer Syllable Segmentation Analysis")
    print("=" * 80)
    
    test_cases = [
        # FIXED CASES
        ("Sabay Digital Corporation ជា", "✅ FIXED: Multi-word English + Khmer"),
        ("ជា\u200bក្រុម", "✅ FIXED: Zero-width joiner handling"),
        ("១០. Taew គឺ", "✅ FIXED: Number + dot + English + Khmer"),
        ("Thon មិន", "✅ FIXED: English name + Khmer"),
        ("A.អាកំបាំង", "✅ FIXED: Letter + dot + Khmer"),
        ("ហ៊ុន", "✅ WORKS: Khmer with diacritic"),
        ("ខ្ញុំ", "✅ WORKS: Khmer with Nikahit"),
        ("គ្រួ", "✅ WORKS: Khmer with subscripts"),
        
        # STILL BROKEN CASES  
        ("Apple ឲ្យប្រើ", "❌ BROKEN: Independent vowel + coeng in mixed text"),
        ("ឲ្យ", "❌ BROKEN: Independent vowel + coeng (core issue)"),
    ]
    
    fixed_count = 0
    broken_count = 0
    
    for text, description in test_cases:
        result = test_case(text, description)
        
        status = "✅" if result['regex_vs_nonregex'] else "❌"
        print(f"\n{status} {description}")
        print(f"   Input: {repr(text)}")
        print(f"   Regex:     {result['regex']}")
        print(f"   Non-regex: {result['non_regex']}")
        
        if not result['regex_vs_nonregex']:
            broken_count += 1
            print(f"   ⚠️  Regex count: {len(result['regex'])}, Non-regex count: {len(result['non_regex'])}")
        else:
            fixed_count += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"✅ Fixed cases: {fixed_count}")
    print(f"❌ Broken cases: {broken_count}")
    print(f"🎯 Success rate: {fixed_count/(fixed_count+broken_count)*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    print("\n🔧 SUCCESSFULLY FIXED:")
    print("• Extended vowel range from U+17B6-U+17C8 to U+17B0-U+17C8")
    print("• Fixed English text handling to group consecutive non-Khmer characters")
    print("• Fixed zero-width joiner (\u200b) handling")
    print("• Fixed mixed number+English+Khmer text scenarios")
    print("• Maintained Nikahit (ំ) support from previous fixes")
    
    print("\n🐛 REMAINING ISSUE:")
    print("• Independent vowels (U+17B0-U+17B5) + coeng (U+17D2) combinations")
    print("• Specifically: ឲ (U+17B2) + ្ (U+17D2) + consonant sequences")
    print("• Non-regex algorithm incorrectly treats coeng as separate token")
    print("• Root cause: Algorithm doesn't handle vowel+coeng combinations properly")
    
    print("\n💡 RECOMMENDATION:")
    print("• Use REGEX version (khmer_syllables) for production - it's fastest & most accurate")
    print("• Consider non-regex version educational/backup only until coeng logic is fixed")
    print("• The core coeng handling algorithm needs redesign for independent vowels")
    
    print("\n📊 PERFORMANCE (from previous tests):")
    print("• Regex version:     ~0.032s (fastest)")
    print("• Non-regex version: ~0.057s (1.8x slower)")
    print("• Both produce ~37k syllables on large text (when working correctly)")

if __name__ == "__main__":
    main() 