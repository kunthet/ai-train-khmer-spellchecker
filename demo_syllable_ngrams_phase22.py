#!/usr/bin/env python3
"""
Demo Script for Phase 2.2: Syllable-Level N-gram Models

This script demonstrates the syllable-level statistical language models
for Khmer spellchecking, building upon the existing syllable segmentation
and character-level models.
"""

import logging
import sys
import pickle
import time
from pathlib import Path
from typing import List

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from statistical_models import (
    SyllableNgramModel,
    SyllableNgramModelTrainer,
    SmoothingMethod
)

from word_cluster.syllable_api import SegmentationMethod


def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_training_data(max_texts: int = 2000) -> List[str]:
    """Load preprocessed training data"""
    
    # Try to load from preprocessed corpus
    corpus_files = list(Path("output").glob("**/corpus_processed_*.pkl"))
    
    if corpus_files:
        print("📂 Loading from preprocessed corpus...")
        latest_file = max(corpus_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_file, 'rb') as f:
                corpus_data = pickle.load(f)
            
            # Extract texts from processing results
            texts = []
            for result in corpus_data:
                if hasattr(result, 'cleaned_text') and result.cleaned_text:
                    texts.append(result.cleaned_text)
                elif hasattr(result, 'original_text') and result.original_text:
                    texts.append(result.original_text)
            
            # Limit to max_texts
            if len(texts) > max_texts:
                texts = texts[:max_texts]
                print(f"📋 Limited to {max_texts} texts for training")
            
            print(f"✅ Loaded {len(texts)} texts from corpus")
            
            # Show sample statistics
            if texts:
                total_chars = sum(len(text) for text in texts)
                avg_length = total_chars / len(texts)
                print(f"📊 Average text length: {avg_length:.1f} characters")
                print(f"📊 Total characters: {total_chars:,}")
                
                # Show sample text
                print(f"📝 Sample text: {texts[0][:100]}...")
            
            return texts
            
        except Exception as e:
            print(f"❌ Error loading corpus: {e}")
            return []
    
    # Fallback to sample texts
    print("📝 Using sample texts...")
    return [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
        "ខ្ញុំស្រលាញ់ភាសាខ្មែរណាស់។",
        "ប្រទេសកម្ពុជាមានទីរដ្ឋធានីភ្នំពេញ។",
        "កម្មវិធីនេះមានប្រយោជន៍ច្រើន។",
        "អាកាសធាតុនេះមានភាពស្រួលសម្រាប់ការធ្វើដំណើរ។",
        "គាត់បានទៅសាលារៀនដើម្បីរៀនភាសាអង់គ្លេស។",
        "ការអប់រំគឺជាគន្លឹះសំខាន់សម្រាប់ការអភិវឌ្ឍន៍។",
        "ប្រជាជនកម្ពុជាមានវប្បធម៌ដ៏រីករាយនិងបុរាណ។"
    ]


def train_syllable_models(texts: List[str]) -> SyllableNgramModelTrainer:
    """
    Train syllable n-gram models
    
    Args:
        texts: Training texts
        
    Returns:
        Trained model trainer
    """
    print(f"\n🧠 TRAINING SYLLABLE N-GRAM MODELS")
    print("=" * 45)
    
    # Initialize trainer with multiple n-gram sizes
    trainer = SyllableNgramModelTrainer(
        ngram_sizes=[2, 3, 4],
        smoothing_method=SmoothingMethod.LAPLACE,
        segmentation_method=SegmentationMethod.REGEX_ADVANCED,
        alpha=1.0,  # Laplace smoothing parameter
        error_threshold=0.001,  # Threshold for error detection
        filter_invalid_syllables=True,  # Filter non-Khmer syllables
        min_syllable_length=1,
        max_syllable_length=8
    )
    
    print(f"🔧 Training configuration:")
    print(f"   N-gram sizes: {trainer.ngram_sizes}")
    print(f"   Smoothing: {trainer.smoothing_method.value}")
    print(f"   Segmentation: {trainer.segmentation_method.value}")
    print(f"   Training texts: {len(texts)}")
    
    # Train all models
    print(f"\n🏋️ Training models...")
    start_time = time.time()
    stats = trainer.train_all_models(texts, show_progress=True)
    training_time = time.time() - start_time
    
    # Display results
    print(f"\n📊 Training Results (completed in {training_time:.1f}s):")
    for n, stat in stats.items():
        print(f"   {n}-gram: {stat.total_ngrams:,} n-grams, {stat.vocabulary_size:,} syllables")
        print(f"            Perplexity: {stat.perplexity:.1f}, Avg syllable length: {stat.avg_syllable_length:.2f}")
    
    return trainer


def test_error_detection(trainer: SyllableNgramModelTrainer):
    """Test syllable-level error detection capabilities"""
    print(f"\n🔍 SYLLABLE ERROR DETECTION TESTING")
    print("=" * 40)
    
    # Test cases with different error types
    test_cases = [
        {
            "text": "នេះជាការសាកល្បងត្រឹមត្រូវ។",
            "description": "Correct Khmer text"
        },
        {
            "text": "នេះជាការសាកល្បងខុស។",
            "description": "Potentially incorrect syllable"
        },
        {
            "text": "ខ្ញុំស្រលាញ់ភាសាខ្មែរណាស់ខុស។",
            "description": "Good text with error at end"
        },
        {
            "text": "ប្រទេសកម្ពុជាមានភាសាខុសៗ។",
            "description": "Mixed correct/incorrect syllables"
        },
        {
            "text": "កម្មវិធីនេះគឺលអ់សម្រាប់រៀន។",
            "description": "Real text with complex syllables"
        }
    ]
    
    print(f"Testing {len(test_cases)} cases with all models...")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        description = test_case["description"]
        
        print(f"📝 Test {i}: {description}")
        print(f"   Text: {text}")
        
        # Run ensemble error detection
        results = trainer.detect_errors_ensemble(text)
        
        for n, result in results.items():
            print(f"   {n}-gram results:")
            print(f"      Syllables: {result.syllables}")
            print(f"      Overall score: {result.overall_score:.3f}")
            print(f"      Errors detected: {len(result.errors_detected)}")
            print(f"      Suspicious syllables: {len(result.suspicious_syllables)}")
            print(f"      Invalid sequences: {len(result.invalid_sequences)}")
            print(f"      Processing time: {result.processing_time*1000:.1f}ms")
            
            # Show detailed errors if any
            if result.errors_detected:
                print(f"      Error details:")
                for start, end, syllable, confidence in result.errors_detected[:3]:
                    print(f"         '{syllable}' at {start}-{end} (confidence: {confidence:.3f})")
        
        print()


def demonstrate_syllable_analysis(trainer: SyllableNgramModelTrainer):
    """Demonstrate syllable-level analysis capabilities"""
    print(f"\n📊 SYLLABLE ANALYSIS DEMONSTRATION")
    print("=" * 40)
    
    # Get a trained model for analysis
    model_3gram = trainer.get_model(3)
    
    print(f"🔍 3-gram Model Analysis:")
    print(f"   Syllable vocabulary: {len(model_3gram.syllable_vocabulary):,} unique syllables")
    print(f"   Total n-grams: {model_3gram.total_ngrams:,}")
    print(f"   Invalid syllables filtered: {len(model_3gram.invalid_syllables):,}")
    
    # Show top syllables
    top_syllables = model_3gram.syllable_frequencies.most_common(10)
    print(f"\n📈 Top 10 most frequent syllables:")
    for i, (syllable, count) in enumerate(top_syllables, 1):
        percentage = (count / sum(model_3gram.syllable_frequencies.values())) * 100
        print(f"   {i:2d}. '{syllable}' ({count:,} times, {percentage:.2f}%)")
    
    # Show syllable length distribution
    print(f"\n📏 Syllable length distribution:")
    for length in sorted(model_3gram.syllable_lengths.keys())[:10]:
        count = model_3gram.syllable_lengths[length]
        percentage = (count / sum(model_3gram.syllable_lengths.values())) * 100
        print(f"   {length} chars: {count:,} syllables ({percentage:.1f}%)")
    
    # Show sample n-grams
    top_ngrams = model_3gram.ngram_counts.most_common(5)
    print(f"\n🔗 Top 5 trigrams:")
    for i, (ngram, count) in enumerate(top_ngrams, 1):
        percentage = (count / model_3gram.total_ngrams) * 100
        print(f"   {i}. '{ngram}' ({count:,} times, {percentage:.3f}%)")


def save_models_and_generate_reports(trainer: SyllableNgramModelTrainer):
    """Save trained models and generate comprehensive reports"""
    print(f"\n💾 SAVING MODELS AND GENERATING REPORTS")
    print("=" * 45)
    
    # Create output directory
    output_dir = Path("output/statistical_models/syllable_ngrams")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all models
    print("💾 Saving models...")
    trainer.save_all_models(str(output_dir))
    
    # Generate training report
    print("📋 Generating training report...")
    training_report = trainer.generate_training_report()
    
    # Save training report
    report_file = output_dir / "syllable_training_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(training_report)
    
    print(f"✅ Training report saved to: {report_file}")
    
    # Display report
    print(f"\n{training_report}")
    
    # Create summary
    summary = {
        "phase": "2.2 - Syllable N-gram Models",
        "models_trained": len(trainer.training_stats),
        "ngram_sizes": list(trainer.training_stats.keys()),
        "smoothing_method": trainer.smoothing_method.value,
        "segmentation_method": trainer.segmentation_method.value,
        "output_directory": str(output_dir),
        "statistics": {}
    }
    
    for n, stats in trainer.training_stats.items():
        summary["statistics"][f"{n}gram"] = {
            "vocabulary_size": stats.vocabulary_size,
            "total_ngrams": stats.total_ngrams,
            "unique_ngrams": stats.unique_ngrams,
            "perplexity": stats.perplexity,
            "avg_syllable_length": stats.avg_syllable_length,
            "syllable_diversity": stats.syllable_diversity
        }
    
    # Save summary
    summary_file = output_dir / "phase22_summary.json"
    import json
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Phase 2.2 summary saved to: {summary_file}")


def compare_with_character_models():
    """Compare syllable models with character models for different use cases"""
    print(f"\n🆚 SYLLABLE vs CHARACTER MODEL COMPARISON")
    print("=" * 45)
    
    comparison_points = [
        {
            "aspect": "Unit of Analysis",
            "syllable": "Syllable-level (រ, ករ, ការ, សាកល្បង)",
            "character": "Character-level (ក, រ, ់, etc.)"
        },
        {
            "aspect": "Vocabulary Size",
            "syllable": "Medium (thousands of syllables)",
            "character": "Small (hundreds of characters)"
        },
        {
            "aspect": "Context Understanding",
            "syllable": "High - understands syllable patterns",
            "character": "Medium - character sequence patterns"
        },
        {
            "aspect": "Error Detection",
            "syllable": "Syllable-level errors and combinations",
            "character": "Character sequence and diacritic errors"
        },
        {
            "aspect": "Processing Speed",
            "syllable": "Fast - fewer units to process",
            "character": "Medium - more units per text"
        },
        {
            "aspect": "Memory Usage",
            "syllable": "Medium - larger vocabulary",
            "character": "Low - smaller vocabulary"
        },
        {
            "aspect": "Best Use Case",
            "syllable": "Word-level errors, grammar checking",
            "character": "Typing errors, diacritic mistakes"
        }
    ]
    
    print(f"{'Aspect':<20} {'Syllable Models':<35} {'Character Models':<35}")
    print("-" * 95)
    
    for point in comparison_points:
        aspect = point["aspect"]
        syllable = point["syllable"]
        character = point["character"]
        print(f"{aspect:<20} {syllable:<35} {character:<35}")
    
    print(f"\n💡 Recommendation: Use both models in ensemble for comprehensive spellchecking!")


def main():
    """Main demo function"""
    setup_logging()
    
    print("🔤 PHASE 2.2: SYLLABLE N-GRAM MODELS DEMO")
    print("=" * 50)
    print("Building syllable-level statistical language models for Khmer spellchecking")
    print()
    
    try:
        # Load training data
        texts = load_training_data(max_texts=2000)
        
        if not texts:
            print("❌ No training data available")
            return
        
        # Train syllable models
        trainer = train_syllable_models(texts)
        
        # Test error detection
        test_error_detection(trainer)
        
        # Demonstrate analysis capabilities
        demonstrate_syllable_analysis(trainer)
        
        # Save models and generate reports
        save_models_and_generate_reports(trainer)
        
        # Compare with character models
        compare_with_character_models()
        
        print(f"\n🎉 PHASE 2.2 COMPLETED SUCCESSFULLY!")
        print("=" * 45)
        print("✅ Syllable n-gram models successfully trained")
        print("✅ Error detection capabilities demonstrated")
        print("✅ Model analysis and statistics generated")
        print("✅ Models saved for production use")
        
        print(f"\n📋 NEXT STEPS:")
        print("   📍 Phase 2.3: Rule-based validation integration")
        print("   📍 Phase 2.4: Statistical model ensemble")
        print("   📍 Phase 3: Neural model enhancement")
        
        print(f"\n📁 OUTPUT FILES:")
        print("   📄 syllable_training_report.txt - Comprehensive training analysis")
        print("   📄 phase22_summary.json - Phase 2.2 summary statistics")
        print("   📄 syllable_*gram_model.json/.pkl - Trained model files")
        
    except Exception as e:
        logging.error(f"Demo failed: {e}", exc_info=True)
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main() 