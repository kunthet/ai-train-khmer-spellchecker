"""
Demo: Model Training Pipeline for Khmer Spellchecker

This demo shows how to use the comprehensive training pipeline to train
all models for the Khmer spellchecker system.
"""

import logging
import time
import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("training_demo")

from train_models import ModelTrainingPipeline


def demo_quick_training():
    """Demo: Quick training with minimal configuration"""
    logger.info("=" * 80)
    logger.info("DEMO 1: Quick Training Configuration")
    logger.info("=" * 80)
    
    # Quick training configuration - reduced models for faster demo
    quick_config = {
        # Data processing
        'max_files': 1,  # Process only 1 file for demo
        'min_khmer_ratio': 0.5,  # More lenient for demo
        'min_text_length': 30,   # Shorter minimum
        'max_text_length': 1000, # Shorter maximum
        'batch_size': 500,       # Smaller batches
        
        # Character N-gram models - only 3-gram for speed
        'character_ngrams': [3],
        'char_smoothing_method': 'laplace',  # Faster than good_turing
        'char_filter_non_khmer': True,
        
        # Syllable N-gram models - only 2-gram for speed
        'syllable_ngrams': [2],
        'syll_smoothing_method': 'laplace',  # Faster than good_turing
        'syll_filter_non_khmer': True,
        
        # Neural models - disabled for quick demo
        'neural_enabled': False,
        
        # Performance settings
        'validate_models': True,
        'detailed_logging': True
    }
    
    try:
        # Check if data directory exists
        data_dirs = [
            "C:/temps/ML-data",
            "/tmp/khmer_data",
            "data",
            "sample_data"
        ]
        
        data_dir = None
        for dir_path in data_dirs:
            if Path(dir_path).exists():
                data_dir = dir_path
                break
        
        if not data_dir:
            logger.warning("No data directory found. Creating sample data...")
            # Create sample data for demo
            sample_dir = Path("sample_data")
            sample_dir.mkdir(exist_ok=True)
            
            sample_texts = [
                "នេះជាការសាកល្បងអត្ថបទខ្មែរសម្រាប់ការបង្វឹកម៉ូដែល។",
                "ការអប់រំជាមូលដ្ឋានសំខាន់សម្រាប់ការអភិវឌ្ឍន៍ប្រទេសជាតិ។",
                "វប្បធម៌ខ្មែរមានប្រវត្តិសាស្ត្រដ៏យូរលង់និងសម្បូរបែប។",
                "ភាសាខ្មែរជាភាសាម្តាយដ៏ស្រស់ស្អាតរបស់ពលរដ្ឋកម្ពុជា។",
                "ការរក្សាប្រទេសជាតិនិងការអភិវឌ្ឍន៍គឺជាកាតព្វកិច្ចរបស់យើងទាំងអស់គ្នា។"
            ] * 50  # Repeat for more training data
            
            sample_file = sample_dir / "sample_khmer_text.txt"
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(sample_texts))
            
            data_dir = str(sample_dir)
            logger.info(f"Created sample data at: {data_dir}")
        
        logger.info(f"Using data directory: {data_dir}")
        
        # Initialize training pipeline
        pipeline = ModelTrainingPipeline(
            data_dir=data_dir,
            output_dir="output/demo_training",
            config=quick_config
        )
        
        # Run training
        start_time = time.time()
        results = pipeline.run_full_training()
        training_time = time.time() - start_time
        
        # Report results
        logger.info("=" * 80)
        logger.info("QUICK TRAINING DEMO RESULTS")
        logger.info("=" * 80)
        
        if results['success']:
            logger.info(f"✅ Quick training completed successfully!")
            logger.info(f"   - Total time: {training_time:.1f}s")
            logger.info(f"   - Models trained: {results['models_trained']}")
            logger.info(f"   - Training texts: {results['training_texts']:,}")
            logger.info(f"   - Output: {results['output_directory']}")
            logger.info(f"   - Validation passed: {results['validation_passed']}")
            return True
        else:
            logger.error(f"❌ Quick training failed: {results['error']}")
            return False
            
    except Exception as e:
        logger.error(f"Quick training demo failed: {e}")
        return False


def demo_full_training():
    """Demo: Full training configuration"""
    logger.info("=" * 80)
    logger.info("DEMO 2: Full Training Configuration")
    logger.info("=" * 80)
    
    # Full training configuration
    full_config = {
        # Data processing
        'max_files': 2,  # Process 2 files for demo
        'min_khmer_ratio': 0.6,
        'min_text_length': 50,
        'max_text_length': 5000,
        'batch_size': 1000,
        
        # Character N-gram models
        'character_ngrams': [3, 4],  # 3-gram and 4-gram
        'char_smoothing_method': 'good_turing',
        'char_filter_non_khmer': True,
        'char_keep_khmer_punctuation': True,
        'char_keep_spaces': True,
        
        # Syllable N-gram models
        'syllable_ngrams': [2, 3],  # 2-gram and 3-gram
        'syll_smoothing_method': 'good_turing',
        'syll_filter_non_khmer': True,
        'syll_min_khmer_ratio': 0.5,
        'syll_filter_multidigit_numbers': True,
        'syll_max_digit_length': 2,
        
        # Neural models
        'neural_enabled': True,
        'neural_sequence_length': 15,  # Shorter for demo
        'neural_vocab_size': 5000,    # Smaller vocab for demo
        'neural_embedding_dim': 64,   # Smaller embedding
        'neural_hidden_dim': 128,     # Smaller hidden
        'neural_num_layers': 2,
        'neural_epochs': 5,           # Fewer epochs for demo
        'neural_batch_size': 16,      # Smaller batch
        'neural_learning_rate': 0.001,
        
        # Ensemble configuration
        'ensemble_neural_weight': 0.35,
        'ensemble_statistical_weight': 0.45,
        'ensemble_rule_weight': 0.20,
        'ensemble_consensus_threshold': 0.65,
        'ensemble_error_confidence_threshold': 0.50,
        
        # Performance settings
        'validate_models': True,
        'save_intermediate': True,
        'detailed_logging': True
    }
    
    try:
        # Check for data directory
        data_dir = "sample_data"  # Use sample data from previous demo
        if not Path(data_dir).exists():
            logger.warning("Sample data not found. Run quick training demo first.")
            return False
        
        # Initialize training pipeline
        pipeline = ModelTrainingPipeline(
            data_dir=data_dir,
            output_dir="output/full_training",
            config=full_config
        )
        
        # Run training
        start_time = time.time()
        results = pipeline.run_full_training()
        training_time = time.time() - start_time
        
        # Report results
        logger.info("=" * 80)
        logger.info("FULL TRAINING DEMO RESULTS")
        logger.info("=" * 80)
        
        if results['success']:
            logger.info(f"✅ Full training completed successfully!")
            logger.info(f"   - Total time: {training_time:.1f}s")
            logger.info(f"   - Models trained: {results['models_trained']}")
            logger.info(f"   - Training texts: {results['training_texts']:,}")
            logger.info(f"   - Output: {results['output_directory']}")
            logger.info(f"   - Validation passed: {results['validation_passed']}")
            
            # Show output structure
            output_dir = Path(results['output_directory'])
            logger.info(f"\n📁 Output Structure:")
            for subdir in ['statistical_models', 'neural_models', 'ensemble_configs']:
                subdir_path = output_dir / subdir
                if subdir_path.exists():
                    files = list(subdir_path.glob('*'))
                    logger.info(f"   {subdir}/ ({len(files)} files)")
                    for file in files[:3]:  # Show first 3 files
                        logger.info(f"     - {file.name}")
                    if len(files) > 3:
                        logger.info(f"     ... and {len(files)-3} more")
            
            return True
        else:
            logger.error(f"❌ Full training failed: {results['error']}")
            return False
            
    except Exception as e:
        logger.error(f"Full training demo failed: {e}")
        return False


def demo_custom_configuration():
    """Demo: Custom configuration from file"""
    logger.info("=" * 80)
    logger.info("DEMO 3: Custom Configuration from File")
    logger.info("=" * 80)
    
    try:
        # Create custom configuration file
        custom_config = {
            "max_files": 1,
            "min_khmer_ratio": 0.4,
            "character_ngrams": [3],
            "syllable_ngrams": [2],
            "neural_enabled": False,
            "char_smoothing_method": "simple_backoff",
            "syll_smoothing_method": "simple_backoff",
            "validate_models": True,
            "detailed_logging": True
        }
        
        config_file = Path("demo_training_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(custom_config, f, indent=2)
        
        logger.info(f"Created custom config file: {config_file}")
        
        # Load and show configuration
        with open(config_file, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        logger.info("Configuration loaded:")
        for key, value in loaded_config.items():
            logger.info(f"  {key}: {value}")
        
        # Initialize pipeline with custom config
        pipeline = ModelTrainingPipeline(
            data_dir="sample_data",
            output_dir="output/custom_training",
            config=loaded_config
        )
        
        # Run training
        results = pipeline.run_full_training()
        
        if results['success']:
            logger.info(f"✅ Custom configuration training completed!")
            logger.info(f"   - Models trained: {results['models_trained']}")
            logger.info(f"   - Training time: {results['total_time']:.1f}s")
            return True
        else:
            logger.error(f"❌ Custom training failed: {results['error']}")
            return False
            
    except Exception as e:
        logger.error(f"Custom configuration demo failed: {e}")
        return False


def main():
    """Run all training demos"""
    logger.info("🚀 STARTING MODEL TRAINING DEMOS")
    logger.info("=" * 90)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    demos = [
        ("Quick Training", demo_quick_training),
        ("Full Training", demo_full_training),
        ("Custom Configuration", demo_custom_configuration)
    ]
    
    results = {}
    total_start_time = time.time()
    
    for demo_name, demo_func in demos:
        logger.info(f"\n🔄 Running {demo_name} Demo...")
        start_time = time.time()
        
        try:
            success = demo_func()
            demo_time = time.time() - start_time
            results[demo_name] = {
                'success': success,
                'time': demo_time
            }
            
            if success:
                logger.info(f"✅ {demo_name} demo completed in {demo_time:.1f}s")
            else:
                logger.error(f"❌ {demo_name} demo failed after {demo_time:.1f}s")
                
        except Exception as e:
            demo_time = time.time() - start_time
            logger.error(f"💥 {demo_name} demo crashed: {e}")
            results[demo_name] = {
                'success': False,
                'time': demo_time,
                'error': str(e)
            }
    
    total_time = time.time() - total_start_time
    
    # Summary
    logger.info("=" * 90)
    logger.info("🏆 TRAINING DEMOS SUMMARY")
    logger.info("=" * 90)
    
    successful_demos = sum(1 for r in results.values() if r['success'])
    total_demos = len(results)
    
    logger.info(f"📊 Results: {successful_demos}/{total_demos} demos successful")
    logger.info(f"⏱️ Total time: {total_time:.1f}s")
    
    for demo_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        logger.info(f"   {status} {demo_name}: {result['time']:.1f}s")
        if not result['success'] and 'error' in result:
            logger.info(f"      Error: {result['error']}")
    
    if successful_demos == total_demos:
        logger.info("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        logger.info("🚀 Ready to train production models with your data!")
    else:
        logger.warning("⚠️ Some demos failed - check error messages above")
    
    return 0 if successful_demos == total_demos else 1


if __name__ == "__main__":
    exit(main()) 