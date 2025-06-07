#!/usr/bin/env python3
"""
Demo Script for Phase 2.4: Statistical Model Ensemble Optimization

This script demonstrates advanced ensemble optimization techniques that improve
upon the basic hybrid validator through sophisticated voting mechanisms,
dynamic weight optimization, and comprehensive performance analysis.
"""

import logging
import sys
import time
import json
from pathlib import Path
from typing import List, Dict

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from statistical_models import (
    EnsembleOptimizer,
    EnsembleConfiguration,
    HybridValidator,
    RuleBasedValidator
)


def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_ensemble_optimization():
    """Test ensemble weight optimization with different methods"""
    print("🔧 ENSEMBLE WEIGHT OPTIMIZATION TESTING")
    print("=" * 45)
    
    # Sample validation texts with various characteristics
    validation_texts = [
        # Pure Khmer texts (good for rule-based validation)
        "នេះជាការសាកល្បងអត្ថបទខ្មែរដែលមានភាពត្រឹមត្រូវ។",
        "ភាសាខ្មែរមានលក្ខណៈពិសេសនៅក្នុងការសរសេរ។",
        "ប្រជាជនកម្ពុជាបានប្រើប្រាស់ភាសានេះអស់រយៈពេលយូរ។",
        
        # Mixed content (challenging for pure rule-based)
        "នេះជាអត្ថបទ 2025 ដែលមានលេខ។",
        "Hello និង ខ្មែរ ភាសាពីរ។",
        "VOA Khmer News ផ្សាយព័ត៌មាន។",
        
        # Texts with potential errors (good test cases)
        "កំ្",  # Invalid coeng usage
        "្ក",   # Coeng at beginning
        "នេះជាអត្ថបទមានបញ្ហា៉ាំ។",  # Multiple diacritics
        
        # Short texts (different model performance)
        "សួរស្តី។",
        "អរគុណ។",
        "បាទ។",
        
        # Complex texts (better for n-gram models)
        "ការអប់រំនៅប្រទេសកម្ពុជាមានភាពចាំបាច់ក្នុងការអភិវឌ្ឍន៍សេដ្ឋកិច្ច។",
        "បច្ចេកវិទ្យាព័ត៌មានបានផ្លាស់ប្តូរวិធីសាស្រ្តនៃការទំនាក់ទំនង។",
        "វប្បធម៌ខ្មែរដែលមានប្រវត្តិសាស្រ្តយូរលង់ត្រូវតែបានអភិរក្ស។"
    ]
    
    print(f"📊 Testing optimization with {len(validation_texts)} validation texts...")
    print()
    
    # Test different optimization configurations
    optimization_configs = [
        {
            'name': 'Cross-Validation (3-fold)',
            'config': EnsembleConfiguration(
                voting_method='weighted',
                enable_weight_optimization=True,
                optimization_method='cross_validation',
                cv_folds=3
            )
        },
        {
            'name': 'Cross-Validation (5-fold)',
            'config': EnsembleConfiguration(
                voting_method='confidence_weighted',
                enable_weight_optimization=True,
                optimization_method='cross_validation',
                cv_folds=5
            )
        },
        {
            'name': 'Dynamic Voting',
            'config': EnsembleConfiguration(
                voting_method='dynamic',
                enable_weight_optimization=True,
                optimization_method='cross_validation',
                cv_folds=3
            )
        }
    ]
    
    optimization_results = []
    
    for config_info in optimization_configs:
        config_name = config_info['name']
        config = config_info['config']
        
        print(f"🔹 Testing {config_name}...")
        
        # Initialize optimizer with this configuration
        optimizer = EnsembleOptimizer(config)
        
        # Optimize weights
        start_time = time.time()
        optimized_weights = optimizer.optimize_ensemble(validation_texts)
        optimization_time = time.time() - start_time
        
        print(f"   ✅ Optimization completed in {optimization_time:.2f}s")
        print(f"   📈 Optimization score: {optimized_weights.optimization_score:.3f}")
        print(f"   ⚖️ Rule weight: {optimized_weights.rule_weight:.3f}")
        print(f"   🔤 Character weights: {optimized_weights.character_weights}")
        print(f"   📝 Syllable weights: {optimized_weights.syllable_weights}")
        
        optimization_results.append({
            'name': config_name,
            'weights': optimized_weights,
            'optimization_time': optimization_time,
            'voting_method': config.voting_method
        })
        print()
    
    return optimization_results, validation_texts


def test_voting_method_comparison(validation_texts: List[str]):
    """Compare different voting methods on the same texts"""
    print("🗳️ VOTING METHOD COMPARISON")
    print("=" * 35)
    
    # Test texts that showcase different voting method strengths
    test_texts = [
        "នេះជាអត្ថបទខ្មែរល្អ។",  # Good Khmer text
        "កំ្",  # Clear error case
        "Hello និង ខ្មែរ",  # Mixed content
        "នេះជាការសាកល្បង២០២៥។",  # Text with numbers
        "្ក",   # Another error case
    ]
    
    voting_methods = ['weighted', 'confidence_weighted', 'dynamic']
    
    print(f"🔍 Comparing {len(voting_methods)} voting methods on {len(test_texts)} test texts...")
    print()
    
    comparison_results = {}
    
    for method in voting_methods:
        print(f"📊 Testing {method} voting...")
        
        # Create configuration for this voting method
        config = EnsembleConfiguration(
            voting_method=method,
            enable_weight_optimization=True,
            optimization_method='cross_validation',
            cv_folds=3
        )
        
        # Initialize and optimize
        optimizer = EnsembleOptimizer(config)
        optimizer.optimize_ensemble(validation_texts)
        
        # Test on sample texts
        method_results = []
        total_processing_time = 0.0
        
        for text in test_texts:
            result = optimizer.validate_text_ensemble(text)
            method_results.append({
                'text': text,
                'overall_score': result.overall_score,
                'confidence_score': result.confidence_score,
                'consensus_score': result.consensus_score,
                'is_valid': result.is_valid,
                'error_count': len(result.errors),
                'processing_time': result.processing_time
            })
            total_processing_time += result.processing_time
        
        # Calculate method statistics
        avg_overall_score = sum(r['overall_score'] for r in method_results) / len(method_results)
        avg_confidence = sum(r['confidence_score'] for r in method_results) / len(method_results)
        avg_consensus = sum(r['consensus_score'] for r in method_results) / len(method_results)
        avg_processing_time = total_processing_time / len(method_results)
        valid_count = sum(1 for r in method_results if r['is_valid'])
        
        comparison_results[method] = {
            'avg_overall_score': avg_overall_score,
            'avg_confidence': avg_confidence,
            'avg_consensus': avg_consensus,
            'avg_processing_time': avg_processing_time,
            'validation_rate': valid_count / len(test_texts),
            'detailed_results': method_results
        }
        
        print(f"   📈 Average overall score: {avg_overall_score:.3f}")
        print(f"   🎯 Average confidence: {avg_confidence:.3f}")
        print(f"   🤝 Average consensus: {avg_consensus:.3f}")
        print(f"   ⚡ Average processing time: {avg_processing_time*1000:.1f}ms")
        print(f"   ✅ Validation rate: {valid_count}/{len(test_texts)} ({valid_count/len(test_texts):.1%})")
        print()
    
    # Determine best method
    best_method = max(comparison_results.keys(), 
                     key=lambda m: comparison_results[m]['avg_overall_score'])
    
    print(f"🏆 Best performing voting method: {best_method}")
    print(f"   Score: {comparison_results[best_method]['avg_overall_score']:.3f}")
    print()
    
    return comparison_results, best_method


def test_performance_evaluation():
    """Test comprehensive performance evaluation"""
    print("📊 COMPREHENSIVE PERFORMANCE EVALUATION")
    print("=" * 40)
    
    # Create comprehensive test dataset
    test_texts = [
        # Correct Khmer texts
        "នេះជាអត្ថបទខ្មែរត្រឹមត្រូវ។",
        "ភាសាខ្មែរមានប្រវត្តិយូរលង់។",
        "ការអប់រំជាមូលដ្ឋានការអភិវឌ្ឍន៍។",
        
        # Mixed content
        "នេះជាអត្ថបទ 2025 ថ្មី។",
        "Hello និង ខ្មែរ language។",
        "Facebook និង social media។",
        
        # Error cases
        "កំ្",  # Coeng error
        "្ក",   # Position error
        "នេះមានបញ្ហា៉ាំ។",  # Diacritic error
        
        # Complex texts
        "បច្ចេកវិទ្យាព័ត៌មានបានផ្លាស់ប្តូរសង្គមសព្វថ្ងៃ។",
        "វប្បធម៌ខ្មែរត្រូវតែបានអភិរក្សដោយជំនាន់ក្រោយ។",
        "សេដ្ឋកិច្ចកម្ពុជាកំពុងមានការលូតលាស់ពីមួយឆ្នាំទៅមួយឆ្នាំ។",
        
        # Short texts
        "បាទ។", "ទេ។", "សួរស្តី។", "អរគុណ។",
        
        # Very long text
        "នេះជាអត្ថបទដែលមានប្រវែងវែងដើម្បីសាកល្បងប្រសិទ្ធភាពការវិភាគនៃប្រព័ន្ធស្ថាបនាសម្រាប់ការពិនិត្យអក្ខរាវិរុទ្ធនិងវេយ្យាករណ៍របស់ភាសាខ្មែរ។"
    ]
    
    print(f"🔍 Evaluating performance on {len(test_texts)} diverse test texts...")
    
    # Test different ensemble configurations
    configurations = [
        ('Basic Hybrid', HybridValidator()),
        ('Optimized Ensemble (weighted)', EnsembleOptimizer(EnsembleConfiguration(
            voting_method='weighted',
            optimization_method='cross_validation',
            cv_folds=3
        ))),
        ('Optimized Ensemble (dynamic)', EnsembleOptimizer(EnsembleConfiguration(
            voting_method='dynamic',
            optimization_method='cross_validation',
            cv_folds=3
        )))
    ]
    
    performance_results = {}
    
    for config_name, validator in configurations:
        print(f"\n📈 Testing {config_name}...")
        
        # Optimize if it's an ensemble optimizer
        if isinstance(validator, EnsembleOptimizer):
            optimizer_training_texts = test_texts[:10]  # Use subset for optimization
            validator.optimize_ensemble(optimizer_training_texts)
        
        # Evaluate performance
        start_time = time.time()
        results = []
        
        for text in test_texts:
            if isinstance(validator, EnsembleOptimizer):
                result = validator.validate_text_ensemble(text)
            else:
                result = validator.validate_text(text)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate comprehensive metrics
        valid_count = sum(1 for r in results if r.is_valid)
        avg_score = sum(r.overall_score for r in results) / len(results)
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        avg_consensus = sum(r.consensus_score for r in results) / len(results)
        avg_processing_time = sum(r.processing_time for r in results) / len(results)
        throughput = len(test_texts) / total_time
        
        total_errors = sum(len(r.errors) for r in results)
        
        performance_results[config_name] = {
            'validation_rate': valid_count / len(test_texts),
            'avg_overall_score': avg_score,
            'avg_confidence_score': avg_confidence,
            'avg_consensus_score': avg_consensus,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'throughput_texts_per_second': throughput,
            'total_errors_detected': total_errors,
            'total_evaluation_time': total_time
        }
        
        print(f"   ✅ Validation rate: {valid_count}/{len(test_texts)} ({valid_count/len(test_texts):.1%})")
        print(f"   📊 Average score: {avg_score:.3f}")
        print(f"   🎯 Average confidence: {avg_confidence:.3f}")
        print(f"   🤝 Average consensus: {avg_consensus:.3f}")
        print(f"   ⚡ Average processing time: {avg_processing_time*1000:.1f}ms")
        print(f"   🚀 Throughput: {throughput:.1f} texts/second")
        print(f"   🔍 Total errors detected: {total_errors}")
    
    # Performance comparison
    print(f"\n🏆 PERFORMANCE COMPARISON")
    print("=" * 25)
    
    best_accuracy = max(performance_results.values(), key=lambda x: x['validation_rate'])
    best_speed = max(performance_results.values(), key=lambda x: x['throughput_texts_per_second'])
    best_confidence = max(performance_results.values(), key=lambda x: x['avg_confidence_score'])
    
    print(f"🎯 Best accuracy: {best_accuracy['validation_rate']:.1%}")
    print(f"⚡ Best speed: {best_speed['throughput_texts_per_second']:.1f} texts/sec")
    print(f"🎯 Best confidence: {best_confidence['avg_confidence_score']:.3f}")
    
    return performance_results


def test_model_integration():
    """Test integration with existing trained models"""
    print("🔗 MODEL INTEGRATION TESTING")
    print("=" * 30)
    
    # Look for existing trained models
    model_dir = Path("output/statistical_models")
    
    print("🔍 Scanning for trained models...")
    
    # Character models
    char_models = {}
    char_model_dirs = ["character_ngrams"]
    for dir_name in char_model_dirs:
        char_dir = model_dir / dir_name
        if char_dir.exists():
            for model_file in char_dir.glob("char_*gram_model.json"):
                n_size = int(model_file.name.split('_')[1].replace('gram', ''))
                char_models[f'character_{n_size}gram'] = str(model_file)
                print(f"   ✅ Found character {n_size}-gram model: {model_file}")
    
    # Syllable models
    syllable_models = {}
    syllable_model_dirs = ["syllable_ngrams"]
    for dir_name in syllable_model_dirs:
        syll_dir = model_dir / dir_name
        if syll_dir.exists():
            for model_file in syll_dir.glob("syllable_*gram_model.json"):
                n_size = int(model_file.name.split('_')[1].replace('gram', ''))
                syllable_models[f'syllable_{n_size}gram'] = str(model_file)
                print(f"   ✅ Found syllable {n_size}-gram model: {model_file}")
    
    # Combine all model paths
    all_model_paths = {**char_models, **syllable_models}
    
    if not all_model_paths:
        print("   ⚠️ No trained models found. Testing with rule-based validation only.")
        
        # Test with rule-based only
        config = EnsembleConfiguration(
            voting_method='weighted',
            optimization_method='cross_validation',
            cv_folds=3
        )
        optimizer = EnsembleOptimizer(config, {})
        
    else:
        print(f"   📊 Found {len(all_model_paths)} trained models")
        print(f"   🔗 Integrating models into ensemble...")
        
        # Create ensemble with loaded models
        config = EnsembleConfiguration(
            voting_method='dynamic',
            optimization_method='cross_validation',
            cv_folds=3,
            character_ngram_sizes=[3, 4, 5],
            syllable_ngram_sizes=[2, 3, 4]
        )
        
        optimizer = EnsembleOptimizer(config, all_model_paths)
    
    # Test ensemble with available models
    test_texts = [
        "នេះជាការសាកល្បងប្រព័ន្ធ។",
        "កំ្",  # Error case
        "Hello និង ខ្មែរ",  # Mixed content
        "្ក",   # Another error case
        "នេះជាអត្ថបទល្អ។"
    ]
    
    print(f"\n🧪 Testing integrated ensemble on {len(test_texts)} texts...")
    
    # Optimize ensemble
    optimizer.optimize_ensemble(test_texts)
    
    # Test validation
    for i, text in enumerate(test_texts, 1):
        result = optimizer.validate_text_ensemble(text)
        
        print(f"\n   📝 Test {i}: '{text}'")
        print(f"      Valid: {'✅' if result.is_valid else '❌'} {result.is_valid}")
        print(f"      Overall score: {result.overall_score:.3f}")
        print(f"      Confidence: {result.confidence_score:.3f}")
        print(f"      Consensus: {result.consensus_score:.3f}")
        print(f"      Errors: {len(result.errors)}")
        
        if result.errors:
            print(f"      Error details:")
            for error in result.errors[:2]:
                print(f"         - [{error.source}] {error.description}")
    
    return len(all_model_paths), optimizer


def test_ensemble_persistence():
    """Test saving and loading optimized ensemble configurations"""
    print("💾 ENSEMBLE PERSISTENCE TESTING")
    print("=" * 35)
    
    # Create and optimize ensemble
    config = EnsembleConfiguration(
        voting_method='dynamic',
        optimization_method='cross_validation',
        cv_folds=3
    )
    
    optimizer = EnsembleOptimizer(config)
    
    validation_texts = [
        "នេះជាការសាកល្បងសម្រាប់ការរក្សាទុក។",
        "កំ្",
        "Hello និង ខ្មែរ",
        "នេះជាអត្ថបទបញ្ចុះបញ្ចូល។"
    ]
    
    print("🔧 Optimizing ensemble configuration...")
    weights = optimizer.optimize_ensemble(validation_texts)
    
    # Save configuration
    output_dir = Path("output/ensemble_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "optimized_ensemble_config.json"
    
    print(f"💾 Saving configuration to: {config_path}")
    optimizer.save_optimized_ensemble(str(config_path))
    
    # Test loading
    print("📂 Loading saved configuration...")
    new_optimizer = EnsembleOptimizer()
    new_optimizer.load_optimized_ensemble(str(config_path))
    
    # Verify loaded configuration
    print(f"✅ Verification:")
    print(f"   Voting method: {new_optimizer.config.voting_method}")
    print(f"   Rule weight: {new_optimizer.optimized_weights.rule_weight:.3f}")
    print(f"   Character weights: {new_optimizer.optimized_weights.character_weights}")
    print(f"   Syllable weights: {new_optimizer.optimized_weights.syllable_weights}")
    print(f"   Optimization score: {new_optimizer.optimized_weights.optimization_score:.3f}")
    
    # Test that loaded ensemble works
    test_text = "នេះជាការសាកល្បងនៃការផ្ទុកឡើងវិញ។"
    result = new_optimizer.validate_text_ensemble(test_text)
    
    print(f"\n🧪 Testing loaded ensemble:")
    print(f"   Text: '{test_text}'")
    print(f"   Valid: {'✅' if result.is_valid else '❌'} {result.is_valid}")
    print(f"   Score: {result.overall_score:.3f}")
    print(f"   Confidence: {result.confidence_score:.3f}")
    
    return config_path


def main():
    """Main demo function for Phase 2.4"""
    setup_logging()
    
    print("🔧 PHASE 2.4: STATISTICAL MODEL ENSEMBLE OPTIMIZATION DEMO")
    print("=" * 70)
    print("Advanced ensemble methods with optimized voting mechanisms and dynamic weight adjustment")
    print()
    
    try:
        # Test 1: Ensemble Weight Optimization
        print("🔹 Step 1: Testing ensemble weight optimization...")
        optimization_results, validation_texts = test_ensemble_optimization()
        
        # Test 2: Voting Method Comparison
        print("🔹 Step 2: Comparing voting methods...")
        voting_comparison, best_method = test_voting_method_comparison(validation_texts)
        
        # Test 3: Performance Evaluation
        print("🔹 Step 3: Comprehensive performance evaluation...")
        performance_results = test_performance_evaluation()
        
        # Test 4: Model Integration
        print("🔹 Step 4: Testing model integration...")
        model_count, integrated_optimizer = test_model_integration()
        
        # Test 5: Ensemble Persistence
        print("🔹 Step 5: Testing ensemble persistence...")
        config_path = test_ensemble_persistence()
        
        # Summary
        print("\n🎉 PHASE 2.4 COMPLETION SUMMARY")
        print("=" * 40)
        
        print("📋 ACHIEVEMENTS:")
        print(f"   ✅ Advanced ensemble optimization with {len(optimization_results)} configurations tested")
        print(f"   ✅ Voting method comparison - best: {best_method}")
        print(f"   ✅ Performance evaluation across {len(performance_results)} configurations")
        print(f"   ✅ Model integration with {model_count} trained models")
        print(f"   ✅ Ensemble persistence and configuration management")
        
        print("\n📊 KEY IMPROVEMENTS:")
        
        # Find best optimization result
        best_optimization = max(optimization_results, key=lambda x: x['weights'].optimization_score)
        print(f"   🏆 Best optimization score: {best_optimization['weights'].optimization_score:.3f}")
        print(f"   🗳️ Best voting method: {best_method}")
        
        # Performance comparison
        if 'Optimized Ensemble (dynamic)' in performance_results:
            dynamic_performance = performance_results['Optimized Ensemble (dynamic)']
            print(f"   ⚡ Dynamic ensemble speed: {dynamic_performance['throughput_texts_per_second']:.1f} texts/sec")
            print(f"   🎯 Dynamic ensemble accuracy: {dynamic_performance['validation_rate']:.1%}")
        
        print("\n🔧 TECHNICAL FEATURES:")
        print("   ✅ Multiple voting mechanisms (weighted, confidence-weighted, dynamic)")
        print("   ✅ Cross-validation weight optimization")
        print("   ✅ Text-characteristic-based dynamic weighting")
        print("   ✅ Enhanced error deduplication across multiple models")
        print("   ✅ Comprehensive performance metrics and analysis")
        print("   ✅ Model persistence and configuration management")
        
        print("\n📍 NEXT STEPS:")
        print("   📍 Phase 3.1: Neural enhancement with character-level LSTM")
        print("   📍 Phase 3.2: Masked language modeling with transformers")
        print("   📍 Phase 3.3: Advanced ensemble with neural components")
        print("   📍 Phase 4: Production API and web interface")
        
        print(f"\n💾 OUTPUTS GENERATED:")
        print(f"   📁 Optimized ensemble configuration: {config_path}")
        print(f"   📊 Performance analysis available in memory")
        print(f"   🔧 Optimized weights for production use")
        
        print("\n🎉 PHASE 2.4 COMPLETED SUCCESSFULLY!")
        print("Advanced ensemble optimization ready for production deployment.")
        
        return True
        
    except Exception as e:
        logging.error(f"Demo failed: {e}", exc_info=True)
        print(f"❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 