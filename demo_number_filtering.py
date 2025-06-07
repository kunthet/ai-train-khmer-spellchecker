#!/usr/bin/env python3
"""
Demo Script for Enhanced Number Filtering

This script demonstrates the new number filtering capabilities that intelligently
handle single digits vs multi-digit numbers, and Khmer vs Arabic numerals.
"""

import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from word_cluster.syllable_frequency_analyzer import SyllableFrequencyAnalyzer, SegmentationMethod


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def test_number_classification():
    """Test the number classification system"""
    print("🔢 NUMBER CLASSIFICATION TEST")
    print("=" * 35)
    
    analyzer = SyllableFrequencyAnalyzer()
    
    test_cases = [
        # Single digits
        "១",      # Single Khmer digit
        "2",      # Single Arabic digit
        "៥",      # Single Khmer digit
        "0",      # Single Arabic digit
        
        # Multi-digit Khmer numbers
        "១២",     # Two Khmer digits
        "២០២៥",   # Four Khmer digits (year)
        "១២៣៤៥", # Five Khmer digits
        
        # Multi-digit Arabic numbers
        "123",    # Three Arabic digits
        "2025",   # Four Arabic digits (year)
        "12345",  # Five Arabic digits
        
        # Mixed digits
        "១2",     # Mixed Khmer-Arabic
        "2០២៥",  # Mixed Arabic-Khmer
        
        # Non-numbers
        "hello",  # English text
        "ការ",    # Khmer syllable
        "a1",     # Mixed text-number
        "12a",    # Mixed number-text
        ".",      # Punctuation
        " ",      # Space
    ]
    
    print("Classification results:")
    print("Syllable".ljust(10) + " | Type".ljust(12) + " | Number Type")
    print("-" * 40)
    
    for syllable in test_cases:
        syllable_type = analyzer._classify_syllable(syllable)
        is_number, number_type = analyzer._is_number_sequence(syllable)
        
        print(f"'{syllable}'".ljust(10) + f" | {syllable_type}".ljust(12) + f" | {number_type if is_number else 'N/A'}")


def test_number_filtering_settings():
    """Test different number filtering configurations"""
    print(f"\n🎛️  NUMBER FILTERING CONFIGURATIONS")
    print("=" * 40)
    
    # Test texts with various number types
    test_texts = [
        "នេះជាការសាកល្បងលេខ ១ និង ២។",              # Single Khmer digits
        "ឆ្នាំ ២០២៥ និងឆ្នាំ ២០២៦។",               # Multi-digit Khmer years
        "Hello world 123 និង 456។",              # Multi-digit Arabic numbers
        "ការប្រកួត World Cup 2026 នឹងធ្វើឡើង។",     # Mixed with Arabic year
        "ទូរស័ព្ទលេខ 012345678 សំរាប់ទំនាក់ទំនង។",   # Phone number
        "គណនា ១+២=៣ និង 4+5=9 ។"                  # Math with mixed digits
    ]
    
    configs = [
        {
            "name": "Allow All Numbers",
            "filter_multidigit_numbers": False,
            "max_digit_length": 10
        },
        {
            "name": "Single Digits Only",
            "filter_multidigit_numbers": True,
            "max_digit_length": 1
        },
        {
            "name": "Up to 2 Digits",
            "filter_multidigit_numbers": True,
            "max_digit_length": 2
        },
        {
            "name": "Filter All Multi-digit",
            "filter_multidigit_numbers": True,
            "max_digit_length": 1
        }
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Settings: filter_multidigit_numbers={config['filter_multidigit_numbers']}, max_digit_length={config['max_digit_length']}")
        
        analyzer = SyllableFrequencyAnalyzer(
            filter_multidigit_numbers=config["filter_multidigit_numbers"],
            max_digit_length=config["max_digit_length"],
            filter_non_khmer=True,
            keep_punctuation=True
        )
        
        stats = analyzer.train_on_texts(test_texts, show_progress=False)
        numbers_filtered = sum(analyzer.number_frequencies.values())
        
        print(f"  Result: {stats.total_syllables} total syllables kept")
        print(f"  Numbers found: {numbers_filtered} number sequences")
        
        # Show top number sequences
        if analyzer.number_frequencies:
            top_numbers = analyzer.number_frequencies.most_common(3)
            numbers_str = ", ".join([f"'{num}'" for num, _ in top_numbers])
            print(f"  Top numbers: {numbers_str}")


def test_khmer_vs_arabic_numbers():
    """Compare handling of Khmer vs Arabic numbers"""
    print(f"\n🇰🇭🆚🌐 KHMER VS ARABIC NUMBERS")
    print("=" * 35)
    
    test_texts = [
        "ឆ្នាំ ២០២៥ ជាឆ្នាំថ្មី។",        # Khmer numbers
        "ឆ្នាំ 2025 ជាឆ្នាំថ្មី។",         # Arabic numbers  
        "លេខ ១២៣ និង ៤៥៦។",           # Khmer numbers
        "លេខ 123 និង 456។",            # Arabic numbers
        "គណនា ១+២=៣ ។",               # Khmer math
        "គណនា 1+2=3 ។",                # Arabic math
        "រាប់ពី ១ ដល់ ១០។",            # Khmer counting
        "រាប់ពី 1 ដល់ 10។"             # Arabic counting
    ]
    
    # Test with aggressive filtering (single digits only)
    analyzer = SyllableFrequencyAnalyzer(
        filter_multidigit_numbers=True,
        max_digit_length=1,
        filter_non_khmer=True
    )
    
    stats = analyzer.train_on_texts(test_texts, show_progress=False)
    
    print("Analysis with single digits only:")
    print(f"Total syllables kept: {stats.total_syllables}")
    print(f"Number sequences found: {sum(analyzer.number_frequencies.values())}")
    
    # Show breakdown by number type
    print("\nNumber sequences by type:")
    for number, freq in analyzer.number_frequencies.most_common():
        is_number, number_type = analyzer._is_number_sequence(number)
        type_desc = {
            'single_khmer': '🇰🇭 Single Khmer',
            'single_arabic': '🌐 Single Arabic',
            'multi_khmer': '🇰🇭 Multi Khmer',
            'multi_arabic': '🌐 Multi Arabic',
            'mixed_digits': '🔄 Mixed'
        }.get(number_type, '❓ Unknown')
        print(f"  '{number}': {freq} ({type_desc})")


def test_real_world_scenarios():
    """Test realistic scenarios with mixed content"""
    print(f"\n🌍 REAL-WORLD SCENARIOS")
    print("=" * 25)
    
    # Realistic mixed content scenarios
    test_texts = [
        "VOA Khmer News ថ្ងៃទី ២៥ ខែមករា ឆ្នាំ ២០២៥។",
        "ការប្រកួតបាល់ទាត់ World Cup 2026 នឹងធ្វើឡើងនៅ USA។",
        "Facebook និង Instagram មានអ្នកប្រើប្រាស់ជាង 1000000 នាក់។",
        "ទូរស័ព្ទ: 012-345-678 ឬអ៊ីមែល: example@gmail.com។",
        "តម្លៃ $15.99 ឬ ១៥.៩៩ ដុល្លារ។",
        "កម្មវិធី AI GPT-4 និង ChatGPT កំពុងពេញនិយម។"
    ]
    
    # Test different filtering strategies
    strategies = [
        {
            "name": "Conservative (Single digits only)",
            "settings": {"filter_multidigit_numbers": True, "max_digit_length": 1}
        },
        {
            "name": "Moderate (Up to 2 digits)",
            "settings": {"filter_multidigit_numbers": True, "max_digit_length": 2}
        },
        {
            "name": "Permissive (Up to 4 digits)",
            "settings": {"filter_multidigit_numbers": True, "max_digit_length": 4}
        }
    ]
    
    for strategy in strategies:
        print(f"\n{strategy['name']}:")
        
        analyzer = SyllableFrequencyAnalyzer(
            filter_non_khmer=True,
            keep_punctuation=True,
            **strategy["settings"]
        )
        
        stats = analyzer.train_on_texts(test_texts, show_progress=False)
        
        print(f"  Total syllables: {stats.total_syllables}")
        print(f"  Unique syllables: {stats.unique_syllables}")
        
        # Show filtering breakdown
        total_before = (sum(analyzer.khmer_syllable_frequencies.values()) + 
                       sum(analyzer.mixed_syllable_frequencies.values()) +
                       sum(analyzer.number_frequencies.values()) +
                       sum(analyzer.non_khmer_syllable_frequencies.values()) +
                       sum(analyzer.punctuation_frequencies.values()))
        
        if total_before > 0:
            filter_ratio = stats.total_syllables / total_before
            print(f"  Filtering ratio: {filter_ratio:.1%} kept")
            
        # Show what numbers were filtered
        if analyzer.number_frequencies:
            filtered_numbers = [num for num in analyzer.number_frequencies.keys()]
            print(f"  Numbers found: {', '.join(filtered_numbers[:5])}")


def generate_comprehensive_report():
    """Generate a comprehensive report with number filtering"""
    print(f"\n📋 COMPREHENSIVE NUMBER FILTERING REPORT")
    print("=" * 45)
    
    # Use diverse test corpus with various number types
    test_texts = [
        "ព្រឹត្តិការណ៍ថ្ងៃទី ១៥ ខែកុម្ភៈ ឆ្នាំ ២០២៥។",
        "ការប្រកួត FIFA World Cup 2026 នៅ North America។", 
        "ប្រជាជនកម្ពុជាមានចំនួន 16000000 នាក់។",
        "ទូរស័ព្ទលេខ 012-345-678 និង 092-123-456។",
        "តម្លៃសម្ភារៈ $12.50 USD ឬ ៥០០០០ រៀល។",
        "កម្មវិធីទូរទស្សន៍ Channel 3, 7, 9 និង 27។",
        "អាស័យដ្ឋាន St. 240, House #25B, Phnom Penh។",
        "ឆ្នាំសិក្សា 2024-2025 នឹងចាប់ផ្តើមខែវិច្ឆិកា។"
    ]
    
    # Use recommended production settings
    analyzer = SyllableFrequencyAnalyzer(
        filter_non_khmer=True,
        keep_punctuation=True,
        filter_multidigit_numbers=True,
        max_digit_length=2  # Allow up to 2-digit numbers
    )
    
    stats = analyzer.train_on_texts(test_texts, show_progress=False)
    
    # Generate and display full report
    report = analyzer.generate_report()
    print(report)
    
    return analyzer


def main():
    """Main demo function"""
    setup_logging()
    
    print("🔢 ENHANCED NUMBER FILTERING DEMO")
    print("=" * 40)
    print("Testing sophisticated number filtering for better Khmer spellchecker models")
    print()
    
    try:
        # Run all tests
        test_number_classification()
        test_number_filtering_settings()
        test_khmer_vs_arabic_numbers()
        test_real_world_scenarios()
        final_analyzer = generate_comprehensive_report()
        
        print(f"\n🎉 NUMBER FILTERING DEMO COMPLETED SUCCESSFULLY!")
        
        print(f"\n📋 Key Benefits of Enhanced Number Filtering:")
        print("   ✅ Intelligent distinction between single vs multi-digit numbers")
        print("   ✅ Separate handling of Khmer vs Arabic numerals")
        print("   ✅ Configurable digit length thresholds")
        print("   ✅ Preserves legitimate numeric content (dates, simple counts)")
        print("   ✅ Filters out noisy long numbers (phone numbers, IDs)")
        print("   ✅ Better frequency models for spellchecking")
        
        print(f"\n🔧 Recommended Settings for Production:")
        print("   • filter_multidigit_numbers=True (enable number filtering)")
        print("   • max_digit_length=2 (allow single/double digits)")
        print("   • Keeps: ១, ២, ១២, 25 (dates, simple counts)")
        print("   • Filters: ២០២៥, 123456 (years, long numbers)")
        print("   • Balance between preserving context and reducing noise")
        
        # Save the enhanced model
        print(f"\n💾 Saving enhanced filtered frequency model...")
        output_dir = Path("output/filtered_models")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / "khmer_enhanced_number_filtered_model"
        final_analyzer.save_frequency_data(str(model_path))
        print(f"   Model saved to: {model_path}.json/.pkl")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main() 