#!/usr/bin/env python3
"""
Demo Script for Syllable Filtering Enhancement

This script demonstrates the new filtering capabilities that categorize
and optionally filter non-Khmer syllables from frequency analysis.
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


def test_filtering_comparison():
    """Compare analysis with and without filtering"""
    print("🔍 SYLLABLE FILTERING COMPARISON DEMO")
    print("=" * 50)
    
    # Sample texts with mixed content
    test_texts = [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
        "VOA Khmer News រាយការណ៍ថ្មី។",
        "ប្រទេសកម្ពុជា Cambodia ២០២៥។",
        "Hello world និង ភាសាខ្មែរ ១២៣។",
        "Phnom Penh ភ្នំពេញ Capital City។",
        "Reuters/BBC News ព័ត៌មានអន្តរជាតិ។",
        "អ្នកសារព័ត៌មាន journalists reporting ការព្រឹត្តិការណ៍។"
    ]
    
    print(f"📝 Test texts ({len(test_texts)} samples):")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")
    print()
    
    # Test WITHOUT filtering
    print("🚫 ANALYSIS WITHOUT FILTERING")
    print("-" * 35)
    
    analyzer_no_filter = SyllableFrequencyAnalyzer(
        filter_non_khmer=False,
        keep_punctuation=True
    )
    
    stats_no_filter = analyzer_no_filter.train_on_texts(test_texts, show_progress=False)
    
    print(f"Total syllables: {stats_no_filter.total_syllables}")
    print(f"Unique syllables: {stats_no_filter.unique_syllables}")
    print("Top 10 syllables:")
    for i, (syll, freq) in enumerate(stats_no_filter.most_common_syllables[:10], 1):
        syll_type = analyzer_no_filter._classify_syllable(syll)
        type_emoji = {"khmer": "🇰🇭", "mixed": "🔄", "punctuation": "📝", "non_khmer": "🌐"}.get(syll_type, "❓")
        print(f"  {i:2d}. '{syll}' {type_emoji}: {freq}")
    
    # Test WITH filtering
    print(f"\n✅ ANALYSIS WITH FILTERING")
    print("-" * 33)
    
    analyzer_with_filter = SyllableFrequencyAnalyzer(
        filter_non_khmer=True,
        keep_punctuation=True,
        min_khmer_ratio=0.5
    )
    
    stats_with_filter = analyzer_with_filter.train_on_texts(test_texts, show_progress=False)
    
    print(f"Total syllables: {stats_with_filter.total_syllables}")
    print(f"Unique syllables: {stats_with_filter.unique_syllables}")
    print("Top 10 syllables:")
    for i, (syll, freq) in enumerate(stats_with_filter.most_common_syllables[:10], 1):
        syll_type = analyzer_with_filter._classify_syllable(syll)
        type_emoji = {"khmer": "🇰🇭", "mixed": "🔄", "punctuation": "📝", "non_khmer": "🌐"}.get(syll_type, "❓")
        print(f"  {i:2d}. '{syll}' {type_emoji}: {freq}")
    
    # Comparison summary
    print(f"\n📊 FILTERING IMPACT SUMMARY")
    print("-" * 29)
    reduction_total = stats_no_filter.total_syllables - stats_with_filter.total_syllables
    reduction_unique = stats_no_filter.unique_syllables - stats_with_filter.unique_syllables
    
    print(f"Total syllables reduced: {reduction_total} ({reduction_total/stats_no_filter.total_syllables:.1%})")
    print(f"Unique syllables reduced: {reduction_unique} ({reduction_unique/stats_no_filter.unique_syllables:.1%})")
    
    # Show categorization breakdown
    print(f"\n📊 SYLLABLE CATEGORIZATION")
    print("-" * 26)
    print(f"🇰🇭 Pure Khmer: {sum(analyzer_with_filter.khmer_syllable_frequencies.values())}")
    print(f"🔄 Mixed content: {sum(analyzer_with_filter.mixed_syllable_frequencies.values())}")
    print(f"📝 Punctuation: {sum(analyzer_with_filter.punctuation_frequencies.values())}")
    print(f"🌐 Non-Khmer (filtered): {sum(analyzer_with_filter.non_khmer_syllable_frequencies.values())}")
    
    return analyzer_with_filter


def test_different_filter_settings():
    """Test different filtering configurations"""
    print(f"\n🎛️  FILTERING CONFIGURATION TEST")
    print("=" * 35)
    
    test_texts = [
        "VOA Khmer News រាយការណ៍ថ្មី។",
        "ប្រទេសកម្ពុជា Cambodia ២០២៥។",
        "Hello world និង ភាសាខ្មែរ ១២៣។"
    ]
    
    configs = [
        {"name": "Strict Khmer Only", "filter_non_khmer": True, "keep_punctuation": False, "min_khmer_ratio": 0.8},
        {"name": "Khmer + Punctuation", "filter_non_khmer": True, "keep_punctuation": True, "min_khmer_ratio": 0.8},
        {"name": "Khmer + Mixed Content", "filter_non_khmer": True, "keep_punctuation": True, "min_khmer_ratio": 0.3},
        {"name": "No Filtering", "filter_non_khmer": False, "keep_punctuation": True, "min_khmer_ratio": 0.0}
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Settings: filter_non_khmer={config['filter_non_khmer']}, keep_punctuation={config['keep_punctuation']}, min_khmer_ratio={config['min_khmer_ratio']}")
        
        analyzer = SyllableFrequencyAnalyzer(
            filter_non_khmer=config["filter_non_khmer"],
            keep_punctuation=config["keep_punctuation"],
            min_khmer_ratio=config["min_khmer_ratio"]
        )
        
        stats = analyzer.train_on_texts(test_texts, show_progress=False)
        print(f"  Result: {stats.total_syllables} total, {stats.unique_syllables} unique syllables")


def test_syllable_validation_with_filtering():
    """Test syllable validation with filtering"""
    print(f"\n🧪 SYLLABLE VALIDATION WITH FILTERING")
    print("=" * 40)
    
    # Train on Khmer texts
    training_texts = [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
        "ខ្ញុំស្រលាញ់ភាសាខ្មែរណាស់។",
        "ប្រទេសកម្ពុជាមានទីរដ្ឋធានីភ្នំពេញ។"
    ]
    
    analyzer = SyllableFrequencyAnalyzer(filter_non_khmer=True, keep_punctuation=True)
    analyzer.train_on_texts(training_texts, show_progress=False)
    
    # Test validation on mixed content
    test_syllables = [
        "នេះ",        # Khmer - should be valid
        "ការ",        # Khmer - should be valid  
        "Hello",      # English - should be unknown
        "២០២៥",      # Khmer numbers - should be valid/mixed
        "។",          # Khmer punctuation - depends on settings
        "VOA",        # English acronym - should be unknown
        "ភ្នំពេញ",     # Khmer compound - should be valid
        "123"         # Numbers - should be unknown
    ]
    
    print("Validation results:")
    for syllable in test_syllables:
        validation = analyzer.validate_syllable(syllable)
        syll_type = analyzer._classify_syllable(syllable)
        type_emoji = {"khmer": "🇰🇭", "mixed": "🔄", "punctuation": "📝", "non_khmer": "🌐"}.get(syll_type, "❓")
        
        print(f"  '{syllable}' {type_emoji}: {validation.rarity_level} (freq: {validation.frequency}, confidence: {validation.confidence_score:.2f})")


def generate_full_report():
    """Generate comprehensive filtering report"""
    print(f"\n📋 COMPREHENSIVE FILTERING REPORT")
    print("=" * 35)
    
    # Use diverse test corpus
    test_texts = [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរដ៏ពិសេស។",
        "VOA Khmer News និង BBC Khmer ផ្សាយព័ត៌មាន។",
        "ការប្រកួតបាល់ទាត់ World Cup ២០២៦ នឹងធ្វើឡើង។",
        "Facebook, Instagram និង TikTok ពេញនិយម។",
        "កម្មវិធី AI (Artificial Intelligence) កំពុងអភិវឌ្ឍ។",
        "ទីក្រុងភ្នំពេញ Phnom Penh Capital City របស់កម្ពុជា។",
        "សាកលវិទ្យាល័យ Harvard University និង MIT ល្បីល្បាញ។",
        "ការបង្ការជំងឺ COVID-19 នៅតែបន្ត។"
    ]
    
    analyzer = SyllableFrequencyAnalyzer(
        filter_non_khmer=True,
        keep_punctuation=True,
        min_khmer_ratio=0.3  # Allow some mixed content
    )
    
    stats = analyzer.train_on_texts(test_texts, show_progress=False)
    
    # Generate and display full report
    report = analyzer.generate_report()
    print(report)
    
    return analyzer


def main():
    """Main demo function"""
    setup_logging()
    
    print("🎯 SYLLABLE FILTERING ENHANCEMENT DEMO")
    print("=" * 45)
    print("Testing filtering capabilities for better Khmer spellchecker models")
    print()
    
    try:
        # Run all tests
        analyzer = test_filtering_comparison()
        test_different_filter_settings()
        test_syllable_validation_with_filtering()
        final_analyzer = generate_full_report()
        
        print(f"\n🎉 FILTERING DEMO COMPLETED SUCCESSFULLY!")
        
        print(f"\n📋 Key Benefits of Filtering:")
        print("   ✅ Cleaner Khmer language models")
        print("   ✅ Better frequency statistics for spellchecking")
        print("   ✅ Reduced noise from mixed-language content")
        print("   ✅ Configurable filtering levels for different use cases")
        print("   ✅ Preserved context information for legitimate mixed content")
        
        print(f"\n🔧 Recommended Settings for Production:")
        print("   • filter_non_khmer=True (enable filtering)")
        print("   • keep_punctuation=True (preserve context)")
        print("   • min_khmer_ratio=0.3 (allow some mixed content)")
        print("   • Filters out pure English/numbers while keeping Khmer-dominant content")
        
        # Save the filtered model
        print(f"\n💾 Saving filtered frequency model...")
        output_dir = Path("output/filtered_models")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / "khmer_filtered_frequency_model"
        final_analyzer.save_frequency_data(str(model_path))
        print(f"   Model saved to: {model_path}.json/.pkl")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main() 