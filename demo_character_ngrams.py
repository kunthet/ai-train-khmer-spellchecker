#!/usr/bin/env python3
"""
Demo Script for Phase 2.1: Character N-gram Models

This script demonstrates the character-level n-gram models for Khmer spellchecking,
including training on real corpus data, error detection, and model evaluation.
"""

import logging
import sys
import json
import pickle
from pathlib import Path
from typing import List

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from statistical_models import (
    NgramModelTrainer,
    CharacterNgramModel,
    SmoothingMethod,
    compare_smoothing_methods
)
from preprocessing.text_pipeline import CorpusProcessor


def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('demo_character_ngrams.log'),
            logging.StreamHandler()
        ]
    )


def load_training_data(max_texts: int = 5000) -> List[str]:
    """
    Load training data from preprocessed corpus
    
    Args:
        max_texts: Maximum number of texts to load
        
    Returns:
        List of training texts
    """
    print("📁 LOADING TRAINING DATA")
    print("=" * 30)
    
    # Try to load existing preprocessed data from both possible locations
    possible_dirs = [
        Path("output/preprocessing"),
        Path("output/demo_preprocessing")
    ]
    
    preprocessed_files = []
    for dir_path in possible_dirs:
        if dir_path.exists():
            preprocessed_files.extend(list(dir_path.glob("corpus_processed_*.pkl")))
    
    if not preprocessed_files:
        print("❌ No preprocessed corpus found. Please run Phase 1.3 preprocessing first.")
        print("   Run: python demo_preprocessing.py")
        return []
    
    # Load the most recent preprocessed file
    latest_file = max(preprocessed_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading preprocessed corpus: {latest_file}")
    
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


def train_ngram_models(texts: List[str]) -> NgramModelTrainer:
    """
    Train character n-gram models
    
    Args:
        texts: Training texts
        
    Returns:
        Trained model trainer
    """
    print(f"\n🧠 TRAINING CHARACTER N-GRAM MODELS")
    print("=" * 40)
    
    # Initialize trainer with multiple n-gram sizes
    trainer = NgramModelTrainer(
        ngram_sizes=[3, 4, 5],
        smoothing_method=SmoothingMethod.LAPLACE,
        alpha=1.0,  # Laplace smoothing parameter
        error_threshold=0.001  # Threshold for error detection
    )
    
    print(f"🔧 Training configuration:")
    print(f"   N-gram sizes: {trainer.ngram_sizes}")
    print(f"   Smoothing: {trainer.smoothing_method.value}")
    print(f"   Training texts: {len(texts)}")
    
    # Train all models
    print(f"\n🏋️ Training models...")
    stats = trainer.train_all_models(texts, show_progress=True)
    
    # Display results
    print(f"\n📊 Training Results:")
    for n, stat in stats.items():
        print(f"   {n}-gram: {stat.total_ngrams:,} n-grams, {stat.unique_ngrams:,} unique")
    
    return trainer


def compare_smoothing_techniques(texts: List[str]):
    """
    Compare different smoothing techniques
    
    Args:
        texts: Sample texts for comparison
    """
    print(f"\n🧮 SMOOTHING TECHNIQUES COMPARISON")
    print("=" * 42)
    
    # Use smaller subset for comparison
    sample_texts = texts[:500] if len(texts) > 500 else texts
    print(f"📊 Using {len(sample_texts)} texts for comparison")
    
    smoothing_methods = [
        (SmoothingMethod.LAPLACE, {"alpha": 1.0}),
        (SmoothingMethod.GOOD_TURING, {"use_simple_gt": True}),
        (SmoothingMethod.SIMPLE_BACKOFF, {"backoff_factor": 0.4})
    ]
    
    # Compare 4-gram models with different smoothing
    results = {}
    
    for method, kwargs in smoothing_methods:
        print(f"\n🔍 Testing {method.value} smoothing...")
        
        # Train model
        model = CharacterNgramModel(
            n=4,
            smoothing_method=method,
            **kwargs
        )
        
        stats = model.train_on_texts(sample_texts, show_progress=False)
        results[method.value] = stats
        
        print(f"   Vocabulary: {stats.vocabulary_size:,} characters")
        print(f"   N-grams: {stats.total_ngrams:,} total, {stats.unique_ngrams:,} unique")
        print(f"   Perplexity: {stats.perplexity:.1f}")
        print(f"   Entropy: {stats.entropy:.3f}")
    
    # Display comparison
    print(f"\n📈 SMOOTHING COMPARISON SUMMARY")
    print("=" * 35)
    print(f"{'Method':<15} {'Perplexity':<12} {'Entropy':<10} {'Coverage':<10}")
    print("-" * 50)
    
    for method_name, stats in results.items():
        print(f"{method_name:<15} {stats.perplexity:<12.1f} {stats.entropy:<10.3f} {stats.coverage:<10.6f}")


def test_error_detection(trainer: NgramModelTrainer):
    """
    Test error detection capabilities
    
    Args:
        trainer: Trained model trainer
    """
    print(f"\n🔍 ERROR DETECTION TESTING")
    print("=" * 32)
    
    # Test cases: correct and incorrect Khmer text
    test_cases = [
        {
            "name": "Correct Khmer",
            "text": "នេះជាការសាកល្បងអត្ថបទខ្មែរត្រឹមត្រូវ។",
            "expected": "few errors"
        },
        {
            "name": "Mixed Content",
            "text": "នេះជា test ការសាកល្បង mixed content។",
            "expected": "some errors"
        },
        {
            "name": "Unusual Sequence",
            "text": "នេះជាការសាកល្បងxxxyyy។",
            "expected": "many errors"
        },
        {
            "name": "Repeated Characters",
            "text": "នេះជាការសាកល្បងគគគគ។",
            "expected": "some errors"
        },
        {
            "name": "Random Characters",
            "text": "abcdxyzqwerty123",
            "expected": "many errors"
        }
    ]
    
    print(f"🧪 Testing {len(test_cases)} cases with ensemble models:")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {case['name']}")
        print(f"   Text: {case['text']}")
        print(f"   Expected: {case['expected']}")
        
        # Run error detection with all models
        results = trainer.detect_errors_ensemble(case['text'])
        
        print(f"   Results:")
        for n, result in results.items():
            error_rate = len(result.errors_detected) / max(len(case['text']), 1)
            print(f"      {n}-gram: {len(result.errors_detected)} errors, "
                  f"score={result.overall_score:.3f}, "
                  f"avg_prob={result.average_probability:.6f}")
        
        # Show most suspicious n-grams
        if results:
            best_result = results[4]  # Use 4-gram results
            if best_result.suspicious_ngrams:
                print(f"   Suspicious 4-grams:")
                for ngram, prob in best_result.suspicious_ngrams[:3]:
                    print(f"      '{ngram}': {prob:.8f}")


def evaluate_model_performance(trainer: NgramModelTrainer, test_texts: List[str]):
    """
    Evaluate model performance on test data
    
    Args:
        trainer: Trained models
        test_texts: Test texts
    """
    print(f"\n📊 MODEL PERFORMANCE EVALUATION")
    print("=" * 38)
    
    if len(test_texts) < 10:
        print("⚠️  Limited test data available for evaluation")
        return
    
    # Use subset for evaluation
    eval_texts = test_texts[:100] if len(test_texts) > 100 else test_texts
    print(f"📋 Evaluating on {len(eval_texts)} texts")
    
    # Evaluate each model
    for n in trainer.ngram_sizes:
        model = trainer.get_model(n)
        print(f"\n🔢 {n}-gram Model Evaluation:")
        
        total_prob = 0.0
        total_errors = 0
        processing_times = []
        
        for text in eval_texts:
            # Get text probability
            text_prob = model.get_text_probability(text)
            total_prob += text_prob
            
            # Run error detection
            result = model.detect_errors(text)
            total_errors += len(result.errors_detected)
            processing_times.append(result.processing_time)
        
        # Calculate metrics
        avg_log_prob = total_prob / len(eval_texts)
        avg_errors = total_errors / len(eval_texts)
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        print(f"   Average log probability: {avg_log_prob:.3f}")
        print(f"   Average errors per text: {avg_errors:.2f}")
        print(f"   Average processing time: {avg_processing_time*1000:.2f}ms")
        
        # Calculate perplexity
        perplexity = math.exp(-avg_log_prob) if avg_log_prob != float('-inf') else float('inf')
        print(f"   Perplexity on test data: {perplexity:.1f}")


def save_trained_models(trainer: NgramModelTrainer):
    """
    Save trained models to disk
    
    Args:
        trainer: Trained models
    """
    print(f"\n💾 SAVING TRAINED MODELS")
    print("=" * 28)
    
    output_dir = Path("output/statistical_models/character_ngrams")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all models
    trainer.save_all_models(str(output_dir))
    
    # Save training report
    report = trainer.generate_training_report()
    report_file = output_dir / "training_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Models saved to: {output_dir}")
    print(f"📋 Training report: {report_file}")
    
    # Save model summary
    summary = {
        "phase": "2.1",
        "component": "character_ngram_models",
        "ngram_sizes": trainer.ngram_sizes,
        "smoothing_method": trainer.smoothing_method.value,
        "model_files": [f"char_{n}gram_model" for n in trainer.ngram_sizes],
        "training_stats": {
            str(n): {
                "vocabulary_size": stats.vocabulary_size,
                "total_ngrams": stats.total_ngrams,
                "unique_ngrams": stats.unique_ngrams,
                "perplexity": stats.perplexity
            }
            for n, stats in trainer.training_stats.items()
        }
    }
    
    summary_file = output_dir / "model_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Model summary: {summary_file}")


def main():
    """Main demo function"""
    setup_logging()
    
    print("📊 PHASE 2.1: CHARACTER N-GRAM MODELS DEMO")
    print("=" * 50)
    print("Building statistical language models for Khmer spellchecking")
    print()
    
    try:
        # Load training data
        texts = load_training_data(max_texts=3000)  # Reasonable size for demo
        
        if not texts:
            print("❌ No training data available. Cannot proceed.")
            return
        
        # Train n-gram models
        trainer = train_ngram_models(texts)
        
        # Compare smoothing techniques
        compare_smoothing_techniques(texts)
        
        # Test error detection
        test_error_detection(trainer)
        
        # Evaluate performance
        evaluate_model_performance(trainer, texts)
        
        # Save models
        save_trained_models(trainer)
        
        # Display final report
        print(f"\n🎉 PHASE 2.1 COMPLETED SUCCESSFULLY!")
        print("=" * 40)
        
        report = trainer.generate_training_report()
        print(report)
        
        print(f"\n📋 Next Steps:")
        print("   ✅ Phase 2.1: Character N-gram Models (COMPLETED)")
        print("   🔄 Phase 2.2: Syllable-Level Statistical Models (NEXT)")
        print("   ⏳ Phase 2.3: Rule-Based Phonological Validation")
        print("   ⏳ Phase 2.4: Dictionary Building System")
        print("   ⏳ Phase 2.5: Edit Distance Correction System")
        
    except Exception as e:
        logging.error(f"Demo failed: {e}", exc_info=True)
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    import math  # Import needed for perplexity calculation
    main() 