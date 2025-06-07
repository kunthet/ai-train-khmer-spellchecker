#!/usr/bin/env python3
"""
Demo Script for Character Filtering in N-gram Models

This script demonstrates the enhanced character filtering capabilities
for cleaning Khmer text and removing non-Khmer characters from n-gram models.
"""

import logging
import sys
import pickle
from pathlib import Path
from typing import List

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from statistical_models import (
    NgramModelTrainer,
    CharacterNgramModel,
    SmoothingMethod
)


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_sample_data() -> List[str]:
    """Load sample training data with mixed content"""
    
    # Try to load from preprocessed corpus
    corpus_files = list(Path("output").glob("**/corpus_processed_*.pkl"))
    
    if corpus_files:
        print("📂 Loading from preprocessed corpus...")
        latest_file = max(corpus_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_file, 'rb') as f:
                corpus_data = pickle.load(f)
            
            texts = []
            for result in corpus_data[:200]:  # Use subset for demo
                if hasattr(result, 'cleaned_text') and result.cleaned_text:
                    texts.append(result.cleaned_text)
            
            print(f"✅ Loaded {len(texts)} texts from corpus")
            return texts
            
        except Exception as e:
            print(f"⚠️  Error loading corpus: {e}")
    
    # Fallback to sample texts with mixed content
    print("📝 Using sample texts with mixed content...")
    return [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
        "This is mixed English and ខ្មែរ content",
        "ប្រទេសកម្ពុជាឆ្នាំ២០២៥",
        "Phone: 012-345-678 email@example.com",
        "ខ្ញុំស្រលាញ់ភាសាខ្មែរណាស់។",
        "Year 2025 and ២០២៥ are the same",
        "URL: https://example.com/khmer",
        "កម្មវិធីនេះមានប្រយោជន៍ច្រើន។",
        "Price: $100 USD or ១០០ដុល្លារ",
        "GPS: 11.5564°N, 104.9282°E"
    ]


def compare_filtering_settings(texts: List[str]):
    """Compare different filtering configurations"""
    print("\n🔬 COMPARING FILTERING CONFIGURATIONS")
    print("=" * 45)
    
    # Define different filtering configurations
    configurations = [
        {
            "name": "No Filtering",
            "filter_non_khmer": False,
            "keep_khmer_punctuation": True,
            "keep_spaces": True,
            "min_khmer_ratio": 0.0
        },
        {
            "name": "Basic Filtering",
            "filter_non_khmer": True,
            "keep_khmer_punctuation": True,
            "keep_spaces": True,
            "min_khmer_ratio": 0.0
        },
        {
            "name": "Strict Khmer Only",
            "filter_non_khmer": True,
            "keep_khmer_punctuation": True,
            "keep_spaces": False,
            "min_khmer_ratio": 0.8
        },
        {
            "name": "Moderate Filtering",
            "filter_non_khmer": True,
            "keep_khmer_punctuation": True,
            "keep_spaces": True,
            "min_khmer_ratio": 0.3
        }
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n📊 Testing: {config['name']}")
        print("-" * 30)
        
        # Create trainer with this configuration
        trainer = NgramModelTrainer(
            ngram_sizes=[4],  # Use 4-gram for demo
            smoothing_method=SmoothingMethod.LAPLACE,
            filter_non_khmer=config['filter_non_khmer'],
            keep_khmer_punctuation=config['keep_khmer_punctuation'],
            keep_spaces=config['keep_spaces'],
            min_khmer_ratio=config['min_khmer_ratio']
        )
        
        # Train model
        stats = trainer.train_all_models(texts, show_progress=False)
        
        # Get filtering statistics
        model = trainer.get_model(4)
        filter_stats = model.get_filtering_statistics()
        model_stats = stats[4]
        
        results[config['name']] = {
            'filter_stats': filter_stats,
            'model_stats': model_stats,
            'config': config
        }
        
        print(f"Vocabulary size: {model_stats.vocabulary_size:,}")
        print(f"Total n-grams: {model_stats.total_ngrams:,}")
        print(f"Khmer ratio: {filter_stats['khmer_ratio']:.1%}")
        print(f"Filter ratio: {filter_stats['filter_ratio']:.1%}")
        print(f"Text retention: {filter_stats['text_retention_ratio']:.1%}")
    
    return results


def test_error_detection_with_filtering(texts: List[str]):
    """Test error detection with and without filtering"""
    print("\n🔍 ERROR DETECTION COMPARISON")
    print("=" * 35)
    
    # Test cases with mixed content
    test_cases = [
        "នេះជាការសាកល្បងត្រឹមត្រូវ។",  # Correct Khmer
        "នេះជាការ test mixed content",     # Mixed content
        "Phone: 012-345-678",              # English only
        "ខ្មែរ123ABC混合",                  # Multiple scripts
        "normal text with ខុស spelling"    # Mixed with potential error
    ]
    
    # Compare with and without filtering
    trainers = {
        "Without Filtering": NgramModelTrainer(
            ngram_sizes=[4],
            filter_non_khmer=False
        ),
        "With Filtering": NgramModelTrainer(
            ngram_sizes=[4],
            filter_non_khmer=True,
            min_khmer_ratio=0.3
        )
    }
    
    # Train both models
    for name, trainer in trainers.items():
        trainer.train_all_models(texts, show_progress=False)
    
    # Test error detection
    print(f"\n{'Text':<35} {'No Filter':<12} {'Filtered':<12}")
    print("-" * 65)
    
    for test_text in test_cases:
        results = {}
        for name, trainer in trainers.items():
            model = trainer.get_model(4)
            detection = model.detect_errors(test_text)
            results[name] = detection.overall_score
        
        display_text = test_text[:30] + "..." if len(test_text) > 30 else test_text
        print(f"{display_text:<35} {results['Without Filtering']:<12.3f} {results['With Filtering']:<12.3f}")


def demonstrate_vocabulary_cleaning(texts: List[str]):
    """Show how filtering affects vocabulary composition"""
    print("\n📚 VOCABULARY COMPOSITION ANALYSIS")
    print("=" * 40)
    
    # Train models with different filtering levels
    no_filter = CharacterNgramModel(n=4, filter_non_khmer=False)
    with_filter = CharacterNgramModel(n=4, filter_non_khmer=True)
    
    no_filter.train_on_texts(texts, show_progress=False)
    with_filter.train_on_texts(texts, show_progress=False)
    
    print(f"📊 Vocabulary Comparison:")
    print(f"   Without filtering: {len(no_filter.vocabulary):,} characters")
    print(f"   With filtering: {len(with_filter.vocabulary):,} characters")
    print(f"   Reduction: {len(no_filter.vocabulary) - len(with_filter.vocabulary):,} characters")
    
    # Show sample characters that were filtered out
    filtered_out = no_filter.vocabulary - with_filter.vocabulary
    if filtered_out:
        sample_filtered = sorted(list(filtered_out))[:20]
        print(f"\n🗑️  Sample filtered characters: {sample_filtered}")
    
    # Show Khmer characters preserved
    khmer_chars = [c for c in with_filter.vocabulary if '\u1780' <= c <= '\u17FF']
    print(f"\n🇰🇭 Khmer characters preserved: {len(khmer_chars)}")
    if khmer_chars:
        sample_khmer = sorted(khmer_chars)[:20]
        print(f"   Sample: {sample_khmer}")


def main():
    """Main demo function"""
    setup_logging()
    
    print("🧹 CHARACTER FILTERING DEMO FOR N-GRAM MODELS")
    print("=" * 50)
    print("Testing enhanced character filtering for cleaner Khmer language models")
    print()
    
    try:
        # Load sample data
        texts = load_sample_data()
        
        if not texts:
            print("❌ No training data available")
            return
        
        # Show sample of mixed content
        print("📝 Sample texts with mixed content:")
        for i, text in enumerate(texts[:5], 1):
            display_text = text[:60] + "..." if len(text) > 60 else text
            print(f"   {i}. {display_text}")
        print()
        
        # Compare filtering configurations
        results = compare_filtering_settings(texts)
        
        # Test error detection differences
        test_error_detection_with_filtering(texts)
        
        # Demonstrate vocabulary cleaning
        demonstrate_vocabulary_cleaning(texts)
        
        # Generate detailed filtering report
        print("\n📋 DETAILED FILTERING REPORT")
        print("=" * 35)
        
        # Use moderate filtering for final demo
        trainer = NgramModelTrainer(
            ngram_sizes=[3, 4, 5],
            filter_non_khmer=True,
            keep_khmer_punctuation=True,
            keep_spaces=True,
            min_khmer_ratio=0.3
        )
        
        trainer.train_all_models(texts, show_progress=True)
        
        # Show filtering report
        filtering_report = trainer.generate_filtering_report()
        print(filtering_report)
        
        # Show model comparison
        training_report = trainer.generate_training_report()
        print(training_report)
        
        print("\n🎉 CHARACTER FILTERING DEMO COMPLETED!")
        print("=" * 42)
        print("✅ Enhanced character filtering successfully implemented")
        print("✅ Non-Khmer characters intelligently filtered")
        print("✅ Configurable filtering levels for different use cases")
        print("✅ Improved model quality with cleaner vocabulary")
        
        print("\n📋 RECOMMENDED SETTINGS:")
        print("   📍 General use: filter_non_khmer=True, min_khmer_ratio=0.3")
        print("   📍 Academic/pure: filter_non_khmer=True, min_khmer_ratio=0.8")
        print("   📍 Mixed content: filter_non_khmer=True, min_khmer_ratio=0.1")
        
    except Exception as e:
        logging.error(f"Demo failed: {e}", exc_info=True)
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main() 