#!/usr/bin/env python3
"""
Phase 2.1: Character N-gram Models Demo

This demo showcases the character-level n-gram models for Khmer spellchecking,
including training, error detection, and model evaluation.
"""

import sys
import pickle
import json
import math
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from statistical_models.character_ngram_model import CharacterNgramModel, NgramModelTrainer
from statistical_models.smoothing_techniques import SmoothingMethod


def load_corpus_data(max_texts: int = 2000):
    """Load preprocessed corpus data"""
    print("📁 LOADING CORPUS DATA")
    print("=" * 25)
    
    # Find preprocessed files
    possible_paths = [
        Path("output/demo_preprocessing/corpus_processed_20250606_230019.pkl"),
        Path("output/preprocessing"),
        Path("output/demo_preprocessing")
    ]
    
    corpus_file = None
    for path in possible_paths:
        if path.is_file():
            corpus_file = path
            break
        elif path.is_dir():
            pkl_files = list(path.glob("corpus_processed_*.pkl"))
            if pkl_files:
                corpus_file = pkl_files[0]
                break
    
    if not corpus_file:
        print("❌ No corpus data found. Using sample texts.")
        return [
            "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
            "ខ្ញុំស្រលាញ់ភាសាខ្មែរណាស់។",
            "ប្រទេសកម្ពុជាមានទីរដ្ឋធានីភ្នំពេញ។",
            "កម្មវិធីនេះមានប្រយោជន៍ច្រើន។",
            "ការសិក្សាភាសាខ្មែរមានសារៈសំខាន់។",
            "យើងកំពុងពិសោធន៍នូវបច្ចេកវិជ្ជាថ្មី។"
        ]
    
    print(f"📂 Loading: {corpus_file}")
    
    try:
        with open(corpus_file, 'rb') as f:
            corpus_data = pickle.load(f)
        
        # Extract texts
        texts = []
        for result in corpus_data[:max_texts]:
            if hasattr(result, 'cleaned_text') and result.cleaned_text:
                texts.append(result.cleaned_text)
        
        print(f"✅ Loaded {len(texts)} texts")
        
        if texts:
            total_chars = sum(len(text) for text in texts)
            print(f"📊 Total characters: {total_chars:,}")
            print(f"📊 Average length: {total_chars/len(texts):.1f} chars")
        
        return texts
        
    except Exception as e:
        print(f"❌ Error loading corpus: {e}")
        return []


def train_models(texts):
    """Train character n-gram models"""
    print(f"\n🧠 TRAINING CHARACTER N-GRAM MODELS")
    print("=" * 40)
    
    # Initialize trainer
    trainer = NgramModelTrainer(
        ngram_sizes=[3, 4, 5],
        smoothing_method=SmoothingMethod.LAPLACE,
        alpha=1.0
    )
    
    print(f"🔧 Configuration:")
    print(f"   N-gram sizes: {trainer.ngram_sizes}")
    print(f"   Smoothing: {trainer.smoothing_method.value}")
    print(f"   Training texts: {len(texts)}")
    
    # Train all models
    print(f"\n🏋️ Training...")
    stats = trainer.train_all_models(texts, show_progress=True)
    
    print(f"\n📊 Results:")
    for n, stat in stats.items():
        print(f"   {n}-gram: {stat.total_ngrams:,} total, {stat.unique_ngrams:,} unique")
        print(f"           Perplexity: {stat.perplexity:.1f}, Entropy: {stat.entropy:.3f}")
    
    return trainer


def test_error_detection(trainer):
    """Test error detection with various inputs"""
    print(f"\n🔍 ERROR DETECTION TESTING")
    print("=" * 32)
    
    test_cases = [
        ("Correct Khmer", "នេះជាការសាកល្បងត្រឹមត្រូវ។"),
        ("Mixed content", "នេះជា test mixed content។"),
        ("Unusual chars", "នេះជាការសាកល្បងxxxyyy។"),
        ("Repeated chars", "នេះជាការសាកល្បងគគគគ។"),
        ("English only", "This is English text only"),
        ("Random chars", "qwerty123xyz"),
    ]
    
    for name, text in test_cases:
        print(f"\n📝 {name}: {text}")
        
        # Test with 4-gram model
        model = trainer.get_model(4)
        result = model.detect_errors(text)
        
        print(f"   Errors: {len(result.errors_detected)}")
        print(f"   Score: {result.overall_score:.3f}")
        print(f"   Avg prob: {result.average_probability:.6f}")
        
        # Show suspicious n-grams
        if result.suspicious_ngrams:
            suspicious = result.suspicious_ngrams[:2]
            suspicious_str = [f"'{ng}': {prob:.8f}" for ng, prob in suspicious]
            print(f"   Suspicious: {suspicious_str}")


def compare_smoothing_methods(texts):
    """Compare different smoothing techniques"""
    print(f"\n🧮 SMOOTHING METHODS COMPARISON")
    print("=" * 38)
    
    # Use subset for comparison
    sample_texts = texts[:300] if len(texts) > 300 else texts
    print(f"📊 Using {len(sample_texts)} texts")
    
    methods = [
        (SmoothingMethod.LAPLACE, {"alpha": 1.0}),
        (SmoothingMethod.GOOD_TURING, {}),
        (SmoothingMethod.SIMPLE_BACKOFF, {"backoff_factor": 0.4})
    ]
    
    print(f"\n{'Method':<15} {'Perplexity':<12} {'Entropy':<10} {'Coverage':<10}")
    print("-" * 50)
    
    for method, kwargs in methods:
        try:
            model = CharacterNgramModel(
                n=4,
                smoothing_method=method,
                **kwargs
            )
            
            stats = model.train_on_texts(sample_texts, show_progress=False)
            
            print(f"{method.value:<15} {stats.perplexity:<12.1f} {stats.entropy:<10.3f} {stats.coverage:<10.6f}")
            
        except Exception as e:
            print(f"{method.value:<15} {'ERROR':<12} {'ERROR':<10} {'ERROR':<10}")


def save_models(trainer):
    """Save trained models"""
    print(f"\n💾 SAVING MODELS")
    print("=" * 18)
    
    output_dir = Path("output/statistical_models/character_ngrams")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    trainer.save_all_models(str(output_dir))
    
    # Save report
    report = trainer.generate_training_report()
    report_file = output_dir / "training_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save summary
    summary = {
        "phase": "2.1",
        "component": "character_ngram_models",
        "ngram_sizes": trainer.ngram_sizes,
        "smoothing": trainer.smoothing_method.value,
        "models": [f"char_{n}gram_model" for n in trainer.ngram_sizes]
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved to: {output_dir}")
    print(f"📋 Report: {report_file.name}")
    print(f"📄 Summary: {summary_file.name}")


def main():
    """Main demo function"""
    print("📊 PHASE 2.1: CHARACTER N-GRAM MODELS")
    print("=" * 45)
    print("Statistical language models for Khmer spellchecking")
    print()
    
    try:
        # Load data
        texts = load_corpus_data(max_texts=2000)
        
        if not texts:
            print("❌ No training data available")
            return
        
        # Train models
        trainer = train_models(texts)
        
        # Test error detection
        test_error_detection(trainer)
        
        # Compare smoothing methods
        compare_smoothing_methods(texts)
        
        # Save models
        save_models(trainer)
        
        # Final report
        print(f"\n🎉 PHASE 2.1 COMPLETED SUCCESSFULLY!")
        print("=" * 38)
        
        print(trainer.generate_training_report())
        
        print(f"\n📋 NEXT STEPS:")
        print("   ✅ Phase 2.1: Character N-gram Models (COMPLETED)")
        print("   🔄 Phase 2.2: Syllable-Level Statistical Models")
        print("   ⏳ Phase 2.3: Rule-Based Phonological Validation") 
        print("   ⏳ Phase 2.4: Dictionary Building System")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 