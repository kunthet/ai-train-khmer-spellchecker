"""
Comprehensive Model Training Pipeline for Khmer Spellchecker

This script trains all models required for the production Khmer spellchecker:
1. Character N-gram models (3-gram, 4-gram, 5-gram)
2. Syllable N-gram models (2-gram, 3-gram, 4-gram)  
3. Syllable-level LSTM neural models
4. Rule-based validation models
5. Ensemble integration configuration

Usage:
    python train_models.py --data_dir /path/to/data --output_dir output/models
"""

import logging
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("model_training")

# Import our modules
from data_collection.file_loader import FileLoader
from preprocessing.text_pipeline import TextPreprocessingPipeline, CorpusProcessor
from preprocessing.statistical_analyzer import StatisticalAnalyzer
from word_cluster.syllable_api import SyllableSegmentationAPI, SegmentationMethod
from word_cluster.syllable_frequency_analyzer import SyllableFrequencyAnalyzer
from statistical_models.character_ngram_models import CharacterNgramModel, NgramModelTrainer
from statistical_models.syllable_ngram_models import SyllableNgramModel, SyllableNgramModelTrainer
from statistical_models.rule_based_validator import RuleBasedValidator
from statistical_models.hybrid_validator import HybridValidator
from statistical_models.ensemble_optimizer import EnsembleOptimizer
from neural_models.syllable_lstm import SyllableLSTMModel, SyllableVocabulary, SyllableLSTMConfiguration
from neural_models.neural_statistical_integration import NeuralStatisticalIntegrator, IntegrationConfiguration


class ModelTrainingPipeline:
    """
    Comprehensive training pipeline for all Khmer spellchecker models
    """
    
    def __init__(self, data_dir: str, output_dir: str = "output/models", config: Dict[str, Any] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config or self._get_default_config()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "statistical_models").mkdir(exist_ok=True)
        (self.output_dir / "neural_models").mkdir(exist_ok=True)
        (self.output_dir / "ensemble_configs").mkdir(exist_ok=True)
        
        # Training statistics
        self.training_stats = {
            'start_time': None,
            'end_time': None,
            'data_stats': {},
            'model_stats': {},
            'errors': []
        }
        
        logger.info(f"Training pipeline initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration"""
        return {
            # Data processing
            'max_files': None,  # Process all files
            'min_khmer_ratio': 0.6,
            'min_text_length': 50,
            'max_text_length': 10000,
            'batch_size': 1000,
            
            # Character N-gram models
            'character_ngrams': [3, 4, 5],
            'char_smoothing_method': 'good_turing',
            'char_filter_non_khmer': True,
            'char_keep_khmer_punctuation': True,
            'char_keep_spaces': True,
            
            # Syllable N-gram models
            'syllable_ngrams': [2, 3, 4],
            'syll_smoothing_method': 'good_turing',
            'syll_filter_non_khmer': True,
            'syll_min_khmer_ratio': 0.5,
            'syll_filter_multidigit_numbers': True,
            'syll_max_digit_length': 2,
            
            # Neural models
            'neural_enabled': True,
            'neural_sequence_length': 20,
            'neural_vocab_size': 8000,
            'neural_embedding_dim': 128,
            'neural_hidden_dim': 256,
            'neural_num_layers': 2,
            'neural_epochs': 10,
            'neural_batch_size': 32,
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
    
    def load_and_preprocess_data(self) -> List[str]:
        """Load and preprocess training data"""
        logger.info("=" * 80)
        logger.info("STEP 1: Loading and Preprocessing Data")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Load files
            logger.info(f"Loading files from {self.data_dir}")
            loader = FileLoader(str(self.data_dir))
            documents = loader.load_all_files(max_files=self.config.get('max_files'))
            
            if not documents:
                raise ValueError(f"No documents found in {self.data_dir}")
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Preprocess texts
            logger.info("Preprocessing texts...")
            pipeline = TextPreprocessingPipeline(
                segmentation_method='regex_advanced',
                min_khmer_ratio=self.config['min_khmer_ratio'],
                min_text_length=self.config['min_text_length'],
                max_text_length=self.config['max_text_length']
            )
            
            all_texts = []
            total_processed = 0
            
            for doc in documents:
                logger.info(f"Processing document: {doc.filename} ({doc.size_mb:.1f}MB)")
                
                # Process document
                results = pipeline.process_document(doc)
                valid_texts = [r.cleaned_text for r in results if r.is_valid]
                all_texts.extend(valid_texts)
                total_processed += len(results)
                
                logger.info(f"  - Processed {len(results)} paragraphs, kept {len(valid_texts)} valid texts")
            
            processing_time = time.time() - start_time
            
            # Calculate statistics
            total_chars = sum(len(text) for text in all_texts)
            avg_length = total_chars / len(all_texts) if all_texts else 0
            
            self.training_stats['data_stats'] = {
                'total_documents': len(documents),
                'total_paragraphs_processed': total_processed,
                'valid_texts': len(all_texts),
                'acceptance_rate': len(all_texts) / total_processed if total_processed > 0 else 0,
                'total_characters': total_chars,
                'average_text_length': avg_length,
                'processing_time': processing_time
            }
            
            logger.info(f"✅ Data preprocessing completed:")
            logger.info(f"   - Total texts: {len(all_texts):,}")
            logger.info(f"   - Total characters: {total_chars:,}")
            logger.info(f"   - Average length: {avg_length:.1f} chars")
            logger.info(f"   - Acceptance rate: {len(all_texts)/total_processed*100:.1f}%")
            logger.info(f"   - Processing time: {processing_time:.1f}s")
            
            return all_texts
            
        except Exception as e:
            error_msg = f"Data preprocessing failed: {e}"
            logger.error(error_msg)
            self.training_stats['errors'].append(error_msg)
            raise
    
    def train_character_ngram_models(self, texts: List[str]) -> Dict[str, str]:
        """Train character n-gram models"""
        logger.info("=" * 80)
        logger.info("STEP 2: Training Character N-gram Models")
        logger.info("=" * 80)
        
        start_time = time.time()
        model_paths = {}
        
        try:
            # Initialize trainer
            trainer = NgramModelTrainer(
                filter_non_khmer=self.config['char_filter_non_khmer'],
                keep_khmer_punctuation=self.config['char_keep_khmer_punctuation'],
                keep_spaces=self.config['char_keep_spaces']
            )
            
            # Train models for each n-gram size
            for n in self.config['character_ngrams']:
                logger.info(f"Training character {n}-gram model...")
                
                model = trainer.train_model(
                    texts=texts,
                    n=n,
                    smoothing_method=self.config['char_smoothing_method']
                )
                
                # Save model
                model_filename = f"character_{n}gram_model"
                json_path = self.output_dir / "statistical_models" / f"{model_filename}.json"
                pkl_path = self.output_dir / "statistical_models" / f"{model_filename}.pkl"
                
                model.save_model(str(json_path), format='json')
                model.save_model(str(pkl_path), format='pickle')
                
                model_paths[f'character_{n}gram'] = {
                    'json': str(json_path),
                    'pickle': str(pkl_path),
                    'vocab_size': len(model.vocab),
                    'total_ngrams': len(model.ngram_counts)
                }
                
                logger.info(f"   ✅ Character {n}-gram model: {len(model.vocab)} vocab, {len(model.ngram_counts):,} n-grams")
            
            training_time = time.time() - start_time
            
            self.training_stats['model_stats']['character_ngrams'] = {
                'models_trained': len(self.config['character_ngrams']),
                'model_paths': model_paths,
                'training_time': training_time
            }
            
            logger.info(f"✅ Character n-gram training completed in {training_time:.1f}s")
            return model_paths
            
        except Exception as e:
            error_msg = f"Character n-gram training failed: {e}"
            logger.error(error_msg)
            self.training_stats['errors'].append(error_msg)
            raise
    
    def train_syllable_ngram_models(self, texts: List[str]) -> Dict[str, str]:
        """Train syllable n-gram models"""
        logger.info("=" * 80)
        logger.info("STEP 3: Training Syllable N-gram Models")
        logger.info("=" * 80)
        
        start_time = time.time()
        model_paths = {}
        
        try:
            # Initialize syllable API
            syllable_api = SyllableSegmentationAPI(SegmentationMethod.REGEX_ADVANCED)
            
            # Segment texts into syllables
            logger.info("Segmenting texts into syllables...")
            syllable_sequences = []
            total_syllables = 0
            
            for i, text in enumerate(texts):
                if i % 1000 == 0:
                    logger.info(f"  Processed {i:,}/{len(texts):,} texts")
                
                result = syllable_api.segment_text(text)
                if result.success:
                    syllable_sequences.append(result.syllables)
                    total_syllables += len(result.syllables)
            
            logger.info(f"Total syllables: {total_syllables:,}")
            
            # Initialize trainer with filtering
            trainer = SyllableNgramModelTrainer(
                filter_non_khmer=self.config['syll_filter_non_khmer'],
                min_khmer_ratio=self.config['syll_min_khmer_ratio'],
                filter_multidigit_numbers=self.config['syll_filter_multidigit_numbers'],
                max_digit_length=self.config['syll_max_digit_length']
            )
            
            # Train models for each n-gram size
            for n in self.config['syllable_ngrams']:
                logger.info(f"Training syllable {n}-gram model...")
                
                model = trainer.train_model(
                    syllable_sequences=syllable_sequences,
                    n=n,
                    smoothing_method=self.config['syll_smoothing_method']
                )
                
                # Save model
                model_filename = f"syllable_{n}gram_model"
                json_path = self.output_dir / "statistical_models" / f"{model_filename}.json"
                pkl_path = self.output_dir / "statistical_models" / f"{model_filename}.pkl"
                
                model.save_model(str(json_path), format='json')
                model.save_model(str(pkl_path), format='pickle')
                
                model_paths[f'syllable_{n}gram'] = {
                    'json': str(json_path),
                    'pickle': str(pkl_path),
                    'vocab_size': len(model.vocab),
                    'total_ngrams': len(model.ngram_counts)
                }
                
                logger.info(f"   ✅ Syllable {n}-gram model: {len(model.vocab)} vocab, {len(model.ngram_counts):,} n-grams")
            
            training_time = time.time() - start_time
            
            self.training_stats['model_stats']['syllable_ngrams'] = {
                'models_trained': len(self.config['syllable_ngrams']),
                'model_paths': model_paths,
                'total_syllables': total_syllables,
                'training_time': training_time
            }
            
            logger.info(f"✅ Syllable n-gram training completed in {training_time:.1f}s")
            return model_paths
            
        except Exception as e:
            error_msg = f"Syllable n-gram training failed: {e}"
            logger.error(error_msg)
            self.training_stats['errors'].append(error_msg)
            raise
    
    def train_neural_models(self, texts: List[str]) -> Dict[str, str]:
        """Train syllable-level LSTM neural models"""
        logger.info("=" * 80)
        logger.info("STEP 4: Training Neural Models")
        logger.info("=" * 80)
        
        if not self.config['neural_enabled']:
            logger.info("Neural training disabled in configuration")
            return {}
        
        start_time = time.time()
        
        try:
            # Initialize syllable API
            syllable_api = SyllableSegmentationAPI(SegmentationMethod.REGEX_ADVANCED)
            
            # Prepare syllable sequences
            logger.info("Preparing syllable sequences for neural training...")
            syllable_sequences = []
            
            for i, text in enumerate(texts):
                if i % 1000 == 0:
                    logger.info(f"  Processed {i:,}/{len(texts):,} texts")
                
                result = syllable_api.segment_text(text)
                if result.success and len(result.syllables) >= 3:  # Minimum length for training
                    syllable_sequences.append(result.syllables)
            
            logger.info(f"Prepared {len(syllable_sequences):,} syllable sequences")
            
            # Build vocabulary
            logger.info("Building syllable vocabulary...")
            vocabulary = SyllableVocabulary()
            vocabulary.build_from_sequences(
                syllable_sequences,
                max_vocab_size=self.config['neural_vocab_size']
            )
            
            # Save vocabulary
            vocab_path = self.output_dir / "neural_models" / "syllable_vocabulary.json"
            vocabulary.save_vocabulary(str(vocab_path))
            
            logger.info(f"Vocabulary size: {vocabulary.vocab_size}")
            
            # Configure model
            config = SyllableLSTMConfiguration(
                vocab_size=vocabulary.vocab_size,
                sequence_length=self.config['neural_sequence_length'],
                embedding_dim=self.config['neural_embedding_dim'],
                hidden_dim=self.config['neural_hidden_dim'],
                num_layers=self.config['neural_num_layers'],
                batch_size=self.config['neural_batch_size'],
                learning_rate=self.config['neural_learning_rate'],
                num_epochs=self.config['neural_epochs']
            )
            
            # Initialize and train model
            logger.info("Training syllable LSTM model...")
            model = SyllableLSTMModel(config, vocabulary)
            
            # Prepare training data
            training_sequences = model.prepare_training_data(syllable_sequences)
            logger.info(f"Training sequences: {len(training_sequences):,}")
            
            # Train model
            training_history = model.train(training_sequences)
            
            # Save model
            model_path = self.output_dir / "neural_models" / "syllable_lstm_model.pth"
            model.save_model(str(model_path))
            
            # Save training history
            history_path = self.output_dir / "neural_models" / "training_history.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(training_history, f, indent=2, ensure_ascii=False)
            
            training_time = time.time() - start_time
            
            model_paths = {
                'model': str(model_path),
                'vocabulary': str(vocab_path),
                'history': str(history_path),
                'vocab_size': vocabulary.vocab_size,
                'training_sequences': len(training_sequences)
            }
            
            self.training_stats['model_stats']['neural'] = {
                'model_path': model_paths,
                'training_time': training_time,
                'epochs': self.config['neural_epochs'],
                'final_loss': training_history['losses'][-1] if training_history['losses'] else None
            }
            
            logger.info(f"✅ Neural model training completed in {training_time:.1f}s")
            return model_paths
            
        except Exception as e:
            error_msg = f"Neural model training failed: {e}"
            logger.error(error_msg)
            self.training_stats['errors'].append(error_msg)
            # Continue without neural models
            logger.warning("Continuing training without neural models")
            return {}
    
    def create_ensemble_configuration(self, char_models: Dict, syll_models: Dict, neural_models: Dict) -> str:
        """Create optimized ensemble configuration"""
        logger.info("=" * 80)
        logger.info("STEP 5: Creating Ensemble Configuration")
        logger.info("=" * 80)
        
        try:
            # Create integration configuration
            integration_config = IntegrationConfiguration(
                neural_weight=self.config['ensemble_neural_weight'],
                statistical_weight=self.config['ensemble_statistical_weight'],
                rule_weight=self.config['ensemble_rule_weight'],
                consensus_threshold=self.config['ensemble_consensus_threshold'],
                error_confidence_threshold=self.config['ensemble_error_confidence_threshold']
            )
            
            # Create ensemble configuration
            ensemble_config = {
                'integration_config': {
                    'neural_weight': integration_config.neural_weight,
                    'statistical_weight': integration_config.statistical_weight,
                    'rule_weight': integration_config.rule_weight,
                    'consensus_threshold': integration_config.consensus_threshold,
                    'error_confidence_threshold': integration_config.error_confidence_threshold,
                    'batch_size': integration_config.batch_size
                },
                'model_paths': {
                    'character_models': char_models,
                    'syllable_models': syll_models,
                    'neural_models': neural_models
                },
                'model_settings': {
                    'character_ngrams': self.config['character_ngrams'],
                    'syllable_ngrams': self.config['syllable_ngrams'],
                    'neural_enabled': bool(neural_models)
                },
                'performance_thresholds': {
                    'min_confidence': 0.3,
                    'max_processing_time': 1.0,
                    'error_rate_threshold': 0.1
                },
                'created_at': datetime.now().isoformat(),
                'training_stats': self.training_stats
            }
            
            # Save configuration
            config_path = self.output_dir / "ensemble_configs" / "production_ensemble_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(ensemble_config, f, indent=2, ensure_ascii=False)
            
            # Also save a simplified version for API usage
            api_config = {
                'neural_weight': integration_config.neural_weight,
                'statistical_weight': integration_config.statistical_weight,
                'rule_weight': integration_config.rule_weight,
                'consensus_threshold': integration_config.consensus_threshold,
                'error_confidence_threshold': integration_config.error_confidence_threshold,
                'model_paths': {
                    'neural_model': neural_models.get('model', ''),
                    'statistical_models': str(self.output_dir / "statistical_models")
                }
            }
            
            api_config_path = self.output_dir / "ensemble_configs" / "api_config.json"
            with open(api_config_path, 'w', encoding='utf-8') as f:
                json.dump(api_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Ensemble configuration saved:")
            logger.info(f"   - Full config: {config_path}")
            logger.info(f"   - API config: {api_config_path}")
            
            return str(config_path)
            
        except Exception as e:
            error_msg = f"Ensemble configuration creation failed: {e}"
            logger.error(error_msg)
            self.training_stats['errors'].append(error_msg)
            raise
    
    def validate_models(self, char_models: Dict, syll_models: Dict, neural_models: Dict) -> Dict[str, Any]:
        """Validate trained models"""
        logger.info("=" * 80)
        logger.info("STEP 6: Model Validation")
        logger.info("=" * 80)
        
        if not self.config['validate_models']:
            logger.info("Model validation disabled")
            return {}
        
        validation_results = {}
        
        try:
            # Test texts for validation
            test_texts = [
                "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
                "ការអប់រំជាមូលដ្ឋានសំខាន់។",
                "វប្បធម៌ខ្មែរមានប្រវត្តិយូរលង់។",
                "កំហុស test error ៕៖។",  # Mixed content with potential errors
                "២០២៥ គឺជាឆ្នាំថ្មី។"  # Numbers and dates
            ]
            
            # Validate character models
            logger.info("Validating character n-gram models...")
            char_validation = {}
            
            for model_name, model_info in char_models.items():
                try:
                    model = CharacterNgramModel.load_model(model_info['pickle'])
                    
                    # Test perplexity calculation
                    perplexities = []
                    for text in test_texts:
                        perplexity = model.calculate_perplexity(text)
                        perplexities.append(perplexity)
                    
                    char_validation[model_name] = {
                        'loaded': True,
                        'vocab_size': model_info['vocab_size'],
                        'total_ngrams': model_info['total_ngrams'],
                        'avg_perplexity': sum(perplexities) / len(perplexities),
                        'test_perplexities': perplexities
                    }
                    
                except Exception as e:
                    char_validation[model_name] = {'loaded': False, 'error': str(e)}
            
            validation_results['character_models'] = char_validation
            
            # Validate syllable models
            logger.info("Validating syllable n-gram models...")
            syll_validation = {}
            
            syllable_api = SyllableSegmentationAPI(SegmentationMethod.REGEX_ADVANCED)
            
            for model_name, model_info in syll_models.items():
                try:
                    model = SyllableNgramModel.load_model(model_info['pickle'])
                    
                    # Test perplexity calculation
                    perplexities = []
                    for text in test_texts:
                        syllables = syllable_api.segment_text(text).syllables
                        perplexity = model.calculate_perplexity(syllables)
                        perplexities.append(perplexity)
                    
                    syll_validation[model_name] = {
                        'loaded': True,
                        'vocab_size': model_info['vocab_size'],
                        'total_ngrams': model_info['total_ngrams'],
                        'avg_perplexity': sum(perplexities) / len(perplexities),
                        'test_perplexities': perplexities
                    }
                    
                except Exception as e:
                    syll_validation[model_name] = {'loaded': False, 'error': str(e)}
            
            validation_results['syllable_models'] = syll_validation
            
            # Validate neural model
            if neural_models:
                logger.info("Validating neural model...")
                try:
                    from neural_models.syllable_lstm import SyllableLSTMModel
                    
                    # Load vocabulary
                    vocabulary = SyllableVocabulary()
                    vocabulary.load_vocabulary(neural_models['vocabulary'])
                    
                    # Create configuration
                    config = SyllableLSTMConfiguration(vocab_size=vocabulary.vocab_size)
                    
                    # Load model
                    model = SyllableLSTMModel(config, vocabulary)
                    model.load_model(neural_models['model'])
                    
                    # Test predictions
                    test_perplexities = []
                    for text in test_texts:
                        syllables = syllable_api.segment_text(text).syllables
                        perplexity = model.calculate_perplexity(syllables)
                        test_perplexities.append(perplexity)
                    
                    validation_results['neural_model'] = {
                        'loaded': True,
                        'vocab_size': vocabulary.vocab_size,
                        'avg_perplexity': sum(test_perplexities) / len(test_perplexities),
                        'test_perplexities': test_perplexities
                    }
                    
                except Exception as e:
                    validation_results['neural_model'] = {'loaded': False, 'error': str(e)}
            
            # Summary
            total_models = len(char_models) + len(syll_models) + (1 if neural_models else 0)
            loaded_models = (
                sum(1 for v in char_validation.values() if v.get('loaded', False)) +
                sum(1 for v in syll_validation.values() if v.get('loaded', False)) +
                (1 if validation_results.get('neural_model', {}).get('loaded', False) else 0)
            )
            
            validation_results['summary'] = {
                'total_models': total_models,
                'loaded_models': loaded_models,
                'success_rate': loaded_models / total_models if total_models > 0 else 0,
                'validation_passed': loaded_models == total_models
            }
            
            logger.info(f"✅ Model validation completed:")
            logger.info(f"   - Total models: {total_models}")
            logger.info(f"   - Successfully loaded: {loaded_models}")
            logger.info(f"   - Success rate: {loaded_models/total_models*100:.1f}%")
            
            return validation_results
            
        except Exception as e:
            error_msg = f"Model validation failed: {e}"
            logger.error(error_msg)
            self.training_stats['errors'].append(error_msg)
            return {'error': error_msg}
    
    def generate_training_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate comprehensive training report"""
        logger.info("=" * 80)
        logger.info("STEP 7: Generating Training Report")
        logger.info("=" * 80)
        
        try:
            # Calculate total training time
            total_time = self.training_stats.get('end_time', time.time()) - self.training_stats.get('start_time', time.time())
            
            report = {
                'training_summary': {
                    'start_time': self.training_stats.get('start_time'),
                    'end_time': self.training_stats.get('end_time'),
                    'total_training_time': total_time,
                    'config_used': self.config,
                    'errors_encountered': len(self.training_stats.get('errors', []))
                },
                'data_statistics': self.training_stats.get('data_stats', {}),
                'model_statistics': self.training_stats.get('model_stats', {}),
                'validation_results': validation_results,
                'output_files': {
                    'statistical_models_dir': str(self.output_dir / "statistical_models"),
                    'neural_models_dir': str(self.output_dir / "neural_models"),
                    'ensemble_configs_dir': str(self.output_dir / "ensemble_configs")
                },
                'errors': self.training_stats.get('errors', [])
            }
            
            # Save detailed report
            report_path = self.output_dir / "training_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            # Generate human-readable summary
            summary_path = self.output_dir / "training_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("KHMER SPELLCHECKER MODEL TRAINING REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total training time: {total_time:.1f} seconds\n\n")
                
                # Data statistics
                data_stats = self.training_stats.get('data_stats', {})
                f.write("DATA STATISTICS:\n")
                f.write(f"  - Documents processed: {data_stats.get('total_documents', 0)}\n")
                f.write(f"  - Valid texts: {data_stats.get('valid_texts', 0):,}\n")
                f.write(f"  - Total characters: {data_stats.get('total_characters', 0):,}\n")
                f.write(f"  - Acceptance rate: {data_stats.get('acceptance_rate', 0)*100:.1f}%\n\n")
                
                # Model statistics
                model_stats = self.training_stats.get('model_stats', {})
                
                if 'character_ngrams' in model_stats:
                    char_stats = model_stats['character_ngrams']
                    f.write("CHARACTER N-GRAM MODELS:\n")
                    f.write(f"  - Models trained: {char_stats.get('models_trained', 0)}\n")
                    f.write(f"  - Training time: {char_stats.get('training_time', 0):.1f}s\n\n")
                
                if 'syllable_ngrams' in model_stats:
                    syll_stats = model_stats['syllable_ngrams']
                    f.write("SYLLABLE N-GRAM MODELS:\n")
                    f.write(f"  - Models trained: {syll_stats.get('models_trained', 0)}\n")
                    f.write(f"  - Total syllables: {syll_stats.get('total_syllables', 0):,}\n")
                    f.write(f"  - Training time: {syll_stats.get('training_time', 0):.1f}s\n\n")
                
                if 'neural' in model_stats:
                    neural_stats = model_stats['neural']
                    f.write("NEURAL MODELS:\n")
                    f.write(f"  - Vocabulary size: {neural_stats.get('model_path', {}).get('vocab_size', 0)}\n")
                    f.write(f"  - Training sequences: {neural_stats.get('model_path', {}).get('training_sequences', 0):,}\n")
                    f.write(f"  - Training time: {neural_stats.get('training_time', 0):.1f}s\n\n")
                
                # Validation results
                if validation_results:
                    summary = validation_results.get('summary', {})
                    f.write("MODEL VALIDATION:\n")
                    f.write(f"  - Total models: {summary.get('total_models', 0)}\n")
                    f.write(f"  - Successfully loaded: {summary.get('loaded_models', 0)}\n")
                    f.write(f"  - Success rate: {summary.get('success_rate', 0)*100:.1f}%\n\n")
                
                # Errors
                errors = self.training_stats.get('errors', [])
                if errors:
                    f.write("ERRORS ENCOUNTERED:\n")
                    for i, error in enumerate(errors, 1):
                        f.write(f"  {i}. {error}\n")
                    f.write("\n")
                
                f.write("Training completed successfully!\n")
                f.write(f"Models saved to: {self.output_dir}\n")
            
            logger.info(f"✅ Training report generated:")
            logger.info(f"   - Detailed report: {report_path}")
            logger.info(f"   - Summary: {summary_path}")
            
            return str(report_path)
            
        except Exception as e:
            error_msg = f"Report generation failed: {e}"
            logger.error(error_msg)
            return ""
    
    def run_full_training(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        logger.info("🚀 STARTING KHMER SPELLCHECKER MODEL TRAINING")
        logger.info("=" * 90)
        
        self.training_stats['start_time'] = time.time()
        
        try:
            # Step 1: Load and preprocess data
            texts = self.load_and_preprocess_data()
            
            if not texts:
                raise ValueError("No valid texts found for training")
            
            # Step 2: Train character n-gram models
            char_models = self.train_character_ngram_models(texts)
            
            # Step 3: Train syllable n-gram models  
            syll_models = self.train_syllable_ngram_models(texts)
            
            # Step 4: Train neural models (if enabled)
            neural_models = self.train_neural_models(texts)
            
            # Step 5: Create ensemble configuration
            ensemble_config = self.create_ensemble_configuration(char_models, syll_models, neural_models)
            
            # Step 6: Validate models
            validation_results = self.validate_models(char_models, syll_models, neural_models)
            
            self.training_stats['end_time'] = time.time()
            
            # Step 7: Generate report
            report_path = self.generate_training_report(validation_results)
            
            # Final summary
            total_time = self.training_stats['end_time'] - self.training_stats['start_time']
            
            logger.info("=" * 90)
            logger.info("🏆 TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 90)
            logger.info(f"✅ Total training time: {total_time:.1f} seconds")
            logger.info(f"✅ Models trained: {len(char_models) + len(syll_models) + (1 if neural_models else 0)}")
            logger.info(f"✅ Training texts: {len(texts):,}")
            logger.info(f"✅ Output directory: {self.output_dir}")
            logger.info(f"✅ Training report: {report_path}")
            
            if validation_results.get('summary', {}).get('validation_passed', False):
                logger.info("🎉 ALL MODELS VALIDATED SUCCESSFULLY - READY FOR PRODUCTION!")
            else:
                logger.warning("⚠️ Some models failed validation - check training report")
            
            return {
                'success': True,
                'total_time': total_time,
                'models_trained': len(char_models) + len(syll_models) + (1 if neural_models else 0),
                'training_texts': len(texts),
                'output_directory': str(self.output_dir),
                'report_path': report_path,
                'validation_passed': validation_results.get('summary', {}).get('validation_passed', False)
            }
            
        except Exception as e:
            self.training_stats['end_time'] = time.time()
            error_msg = f"Training pipeline failed: {e}"
            logger.error(error_msg)
            
            # Generate error report
            try:
                self.generate_training_report({})
            except:
                pass
            
            return {
                'success': False,
                'error': error_msg,
                'total_time': self.training_stats['end_time'] - self.training_stats['start_time'],
                'errors': self.training_stats.get('errors', [])
            }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Khmer Spellchecker Models")
    parser.add_argument("--data_dir", required=True, help="Directory containing training data")
    parser.add_argument("--output_dir", default="output/models", help="Output directory for trained models")
    parser.add_argument("--config", help="Path to training configuration JSON file")
    parser.add_argument("--no_neural", action="store_true", help="Disable neural model training")
    parser.add_argument("--max_files", type=int, help="Maximum number of files to process")
    parser.add_argument("--quick", action="store_true", help="Quick training with reduced parameters")
    
    args = parser.parse_args()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")
    
    # Apply command line overrides
    if config is None:
        config = {}
    
    if args.no_neural:
        config['neural_enabled'] = False
        
    if args.max_files:
        config['max_files'] = args.max_files
        
    if args.quick:
        # Quick training configuration
        config.update({
            'character_ngrams': [3],
            'syllable_ngrams': [2],
            'neural_epochs': 3,
            'neural_batch_size': 16
        })
        logger.info("Using quick training configuration")
    
    # Initialize and run training pipeline
    try:
        pipeline = ModelTrainingPipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config
        )
        
        results = pipeline.run_full_training()
        
        if results['success']:
            print(f"\n🎉 Training completed successfully!")
            print(f"📊 Models trained: {results['models_trained']}")
            print(f"⏱️ Total time: {results['total_time']:.1f}s") 
            print(f"📁 Output: {results['output_directory']}")
            print(f"📄 Report: {results['report_path']}")
            return 0
        else:
            print(f"\n❌ Training failed: {results['error']}")
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n💥 Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 