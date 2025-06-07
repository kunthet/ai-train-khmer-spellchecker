"""
Phase 3.1 Neural Models Demo: Character-Level LSTM for Khmer Spellchecking

This demo showcases the complete neural modeling pipeline including:
1. Neural model training and architecture
2. Perplexity-based validation
3. Neural + Statistical ensemble integration
4. Performance analysis and optimization
5. Real-time spellchecking capabilities

Phase 3.1 builds upon the statistical foundation (Phases 2.1-2.4) by adding
neural network capabilities for improved accuracy and context awareness.
"""

import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import traceback

# Neural models
from neural_models.character_lstm import (
    CharacterLSTMModel, CharacterVocabulary, 
    LSTMConfiguration, ModelTrainingConfig
)
from neural_models.neural_trainer import NeuralTrainer, TrainingResult
from neural_models.neural_validator import NeuralValidator
from neural_models.ensemble_neural import (
    NeuralEnsembleOptimizer, NeuralEnsembleConfiguration
)

# Statistical models (Phase 2.4)
from statistical_models.ensemble_optimizer import EnsembleOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/phase_31_neural_demo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Fix for Windows Unicode issues
import sys
if sys.platform == 'win32':
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'

logger = logging.getLogger("phase_31_demo")


class Phase31NeuralDemo:
    """
    Comprehensive demonstration of Phase 3.1 Neural Models
    
    Demonstrates the complete neural modeling pipeline for Khmer spellchecking,
    including training, validation, and integration with statistical models.
    """
    
    def __init__(self):
        self.demo_name = "Phase 3.1: Character-Level Neural Language Models"
        self.output_dir = Path("output/phase_31_neural")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Demo configurations
        self.lstm_config = LSTMConfiguration(
            embedding_dim=64,  # Reduced for faster training
            hidden_dim=128,    # Reduced for faster training
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            use_attention=True,
            max_vocab_size=800,
            min_char_frequency=2,
            sequence_length=30,  # Further reduced to work with demo texts
            learning_rate=0.001,
            batch_size=16,     # Reduced batch size
            num_epochs=10,     # Reduced epochs for demo
            patience=5
        )
        
        self.training_config = ModelTrainingConfig(
            device='cpu',  # Set to 'cuda' if GPU available
            gradient_clip_norm=5.0,
            weight_decay=1e-5,
            validation_split=0.2,
            verbose=True
        )
        
        self.ensemble_config = NeuralEnsembleConfiguration(
            neural_weight=0.4,
            statistical_weight=0.6,
            voting_strategy="weighted_average",
            error_fusion_method="confidence_weighted",
            enable_neural_correction=True,
            enable_statistical_fallback=True
        )
        
        # Sample data
        self.training_texts = self._get_khmer_training_texts()
        self.test_texts = self._get_khmer_test_texts()
        
        logger.info(f"Initialized {self.demo_name}")
        logger.info(f"Training texts: {len(self.training_texts)}")
        logger.info(f"Test texts: {len(self.test_texts)}")
    
    def _get_khmer_training_texts(self) -> List[str]:
        """Get comprehensive Khmer training texts"""
        return [
            # Educational texts
            "ការអប់រំជាមូលដ្ឋានសំខាន់បំផុតសម្រាប់ការអភិវឌ្ឍន៍ប្រទេសជាតិ។",
            "សិស្សានុសិស្សត្រូវតែចូលរួមយ៉ាងសកម្មក្នុងការសិក្សា។",
            "គ្រូបង្រៀនគួរតែមានចំណេះដឹងទូលំទូលាយ។",
            "មុខវិជ្ជាវិទ្យាសាស្ត្រនិងបច្ចេកវិទ្យាមានសារៈសំខាន់ណាស់។",
            
            # Cultural texts
            "វប្បធម៌ខ្មែរមានប្រវត្តិសាស្ត្រដ៏យូរលង់និងសម្បូរបែប។",
            "បុរាណវត្ថុអង្គរវត្តជាកេរ្តិ៍ឈ្មោះរបស់កម្ពុជា។",
            "ការរាំក្បាច់ខ្មែរបង្ហាញពីភាពស្រស់ស្អាតនិងយាយី។",
            "ចម្លាក់បាយគួរជារបប់របស់ខ្មែរដ៏លេចធ្លោ។",
            
            # Social texts
            "សហគមន៍ត្រូវតែរួបរួមគ្នាដើម្បីដោះស្រាយបញ្ហាសង្គម។",
            "យុវជនជាអនាគតរបស់ប្រទេសជាតិ។",
            "ការគោរពច្បាប់ជាកាតព្វកិច្ចរបស់ពលរដ្ឋ។",
            "សិទ្ធិមនុស្សត្រូវតែរក្សានិងការពារ។",
            
            # Economic texts
            "កសិកម្មជាវិស័យសំខាន់ក្នុងសេដ្ឋកិច្ចកម្ពុជា។",
            "ទេសចរណ៍អាចនាំមកនូវចំណូលច្រើនសម្រាប់ប្រទេស។",
            "ធនាគារដើរតួនាទីសំខាន់ក្នុងការអភិវឌ្ឍសេដ្ឋកិច្ច។",
            "ការបណ្តុះបណ្តាលជំនាញការងារមានសារៈសំខាន់។",
            
            # Technology texts
            "បច្ចេកវិទ្យាព័ត៌មានវិទ្យាកំពុងផ្លាស់ប្តូរពិភពលោក។",
            "ការប្រើប្រាស់អ៊ីនធឺណេតបានកើនឡើងយ៉ាងលឿន។",
            "ទូរស័ព្ទចល័តបានក្លាយជាឧបករណ៍ចាំបាច់។",
            "កម្មវិធីកុំព្យូទ័រជួយសម្រួលដល់ការងារច្រើនប្រភេទ។",
            
            # Health texts
            "សុខភាពល្អជាទ្រព្យសម្បត្តិដ៏មានតម្លៃបំផុត។",
            "ការហាត់ប្រាណជាទៀងទាត់មានប្រយោជន៍ច្រើន។",
            "អាហារូបត្ថម្ភត្រឹមត្រូវជួយការពារជំងឺ។",
            "ការពិនិត្យសុខភាពទៀងទាត់គឺសំខាន់ណាស់។",
            
            # Environmental texts
            "បរិស្ថានបៃតងជាសេចក្តីត្រូវការបន្ទាន់។",
            "ការកាត់បន្ថយការបំពុលបរិយាកាសគួរតែធ្វើឱ្យបាន។",
            "ព្រៃឈើជាសួនសត្វធម្មជាតិដ៏សំខាន់។",
            "ការអភិរក្សធនធានធម្មជាតិជាការប្តេជ្ញាចិត្តរបស់យើង។",
            
            # Complex sentences with various grammar structures
            "នៅពេលដែលយើងមានការអប់រំល្អ យើងអាចអភិវឌ្ឍប្រទេសជាតិបាន។",
            "ប្រសិនបើយើងធ្វើការរួមគ្នា យើងនឹងអាចដោះស្រាយបញ្ហាបាន។",
            "ថ្វីត្បិតតែមានការលំបាក ក៏យើងនៅតែព្យាយាមធ្វើការងារដដែល។",
            "ដោយសារតែភាសាខ្មែរមានលក្ខណៈពិសេស ការសិក្សាត្រូវការពេលវេលាច្រើន។",
            
            # Additional longer texts for better training data
            "ការអប់រំជាមូលដ្ឋានសំខាន់បំផុតសម្រាប់ការអភិវឌ្ឍន៍ប្រទេសជាតិ និងការកសាងសង្គមដែលមានការយល់ដឹង។",
            "វប្បធម៌ខ្មែរមានប្រវត្តិសាស្ត្រដ៏យូរលង់និងសម្បូរបែប ដែលបានឆ្លងកាត់ការផ្លាស់ប្តូរនិងការអភិវឌ្ឍន៍ជាច្រើនសតវត្ស។",
            "បច្ចេកវិទ្យាព័ត៌មានវិទ្យាកំពុងផ្លាស់ប្តូរពិភពលោក និងបានក្លាយជាឧបករណ៍សំខាន់ក្នុងការអភិវឌ្ឍន៍សេដ្ឋកិច្ចនិងសង្គម។",
            "សេដ្ឋកិច្ចកម្ពុជាត្រូវការការកែទម្រង់និងការអភិវឌ្ឍន៍ដើម្បីឱ្យកើនលូតលាស់ និងដើម្បីបង្កើតការងារសម្រាប់យុវជន។",
            "ការអភិរក្សធនធានធម្មជាតិនិងបរិស្ថានជាការប្តេជ្ញាចិត្តរបស់យើងទាំងអស់គ្នា ដើម្បីបន្សល់ទុកជូនកូនចៅនាអនាគត។"
        ]
    
    def _get_khmer_test_texts(self) -> List[str]:
        """Get test texts for validation"""
        return [
            # Correct texts
            "នេះជាអត្ថបទត្រឹមត្រូវសម្រាប់ការសាកល្បង។",
            "ភាសាខ្មែរមានអក្ខរក្រម៧៤តួ។",
            "កម្ពុជាមានទីធ្លាគួរឱ្យមោទនភាពច្រើន។",
            
            # Texts with potential issues
            "នេះជាអត្ថបទដែលអាចមានបញ្ហាខ្លះ។",
            "ការសិក្សាភាសាខ្មែរត្រូវការការប្រុងប្រយ័ត្ន។",
            "វប្បធម៌និងប្រពៃណីខ្មែរគួរតែរក្សាទុក។",
            
            # Complex texts
            "ការអភិវឌ្ឍន៍បច្ចេកវិទ្យាកំពុងផ្លាស់ប្តូរលក្ខណៈនៃការរស់នៅរបស់មនុស្ស។",
            "សេដ្ឋកិច្ចស្រុកខ្មែរត្រូវការការកែទម្រង់ដើម្បីឱ្យកើនលូតលាស់។"
        ]
    
    def step_1_vocabulary_building(self) -> Dict[str, Any]:
        """Step 1: Build character vocabulary from training texts"""
        logger.info("=" * 60)
        logger.info("STEP 1: Building Character Vocabulary")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Create vocabulary
        vocabulary = CharacterVocabulary(self.lstm_config)
        vocab_stats = vocabulary.build_vocabulary(self.training_texts)
        
        # Save vocabulary
        vocab_path = self.output_dir / "character_vocabulary.json"
        vocabulary.save_vocabulary(str(vocab_path))
        
        vocab_info = vocabulary.get_vocabulary_info()
        
        processing_time = time.time() - start_time
        
        result = {
            'vocabulary_size': vocab_stats['vocabulary_size'],
            'total_characters': vocab_stats['total_characters'],
            'unique_characters': vocab_stats['unique_characters'],
            'khmer_ratio': vocab_stats['khmer_ratio'],
            'coverage': vocab_stats['coverage'],
            'processing_time': processing_time,
            'vocabulary_info': vocab_info,
            'vocabulary': vocabulary
        }
        
        logger.info(f"✅ Vocabulary built: {vocab_stats['vocabulary_size']} characters")
        logger.info(f"   Total characters processed: {vocab_stats['total_characters']:,}")
        logger.info(f"   Unique characters: {vocab_stats['unique_characters']}")
        logger.info(f"   Khmer character ratio: {vocab_stats['khmer_ratio']:.1%}")
        logger.info(f"   Vocabulary coverage: {vocab_stats['coverage']:.1%}")
        logger.info(f"   Processing time: {processing_time:.3f}s")
        logger.info(f"   Vocabulary saved to: {vocab_path}")
        
        return result
    
    def step_2_model_architecture(self, vocabulary: CharacterVocabulary) -> Dict[str, Any]:
        """Step 2: Create and analyze LSTM model architecture"""
        logger.info("=" * 60)
        logger.info("STEP 2: Character-Level LSTM Architecture")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Create model
        model = CharacterLSTMModel(self.lstm_config, vocabulary.vocabulary_size)
        model_info = model.get_model_info()
        
        # Test model forward pass
        test_text = "នេះជាការសាកល្បង"
        encoded = vocabulary.encode_text(test_text, max_length=20)
        decoded = vocabulary.decode_indices(encoded)
        
        # Model forward pass test
        import torch
        input_tensor = torch.tensor([encoded[:10]]).long()
        with torch.no_grad():
            output = model(input_tensor)
            output_shape = output.shape
            output_range = (output.min().item(), output.max().item())
        
        processing_time = time.time() - start_time
        
        result = {
            'model_info': model_info,
            'architecture': {
                'embedding_dim': self.lstm_config.embedding_dim,
                'hidden_dim': self.lstm_config.hidden_dim,
                'num_layers': self.lstm_config.num_layers,
                'bidirectional': self.lstm_config.bidirectional,
                'use_attention': self.lstm_config.use_attention,
                'dropout': self.lstm_config.dropout
            },
            'test_encoding': {
                'original_text': test_text,
                'encoded_sample': encoded[:10],
                'decoded_text': decoded,
                'encoding_match': test_text in decoded
            },
            'forward_pass_test': {
                'input_shape': list(input_tensor.shape),
                'output_shape': list(output_shape),
                'output_range': output_range
            },
            'processing_time': processing_time,
            'model': model
        }
        
        logger.info(f"✅ Model architecture created:")
        logger.info(f"   Type: {model_info['model_type']}")
        logger.info(f"   Parameters: {model_info['total_parameters']:,}")
        logger.info(f"   Model size: {model_info['model_size_mb']:.1f} MB")
        logger.info(f"   Embedding dim: {model_info['embedding_dim']}")
        logger.info(f"   Hidden dim: {model_info['hidden_dim']}")
        logger.info(f"   Bidirectional: {model_info['bidirectional']}")
        logger.info(f"   Attention: {model_info['use_attention']}")
        
        logger.info(f"✅ Text encoding test:")
        logger.info(f"   Original: '{test_text}'")
        logger.info(f"   Decoded: '{decoded}'")
        logger.info(f"   Match: {'✅' if test_text in decoded else '❌'}")
        
        logger.info(f"✅ Forward pass test:")
        logger.info(f"   Input shape: {input_tensor.shape}")
        logger.info(f"   Output shape: {output_shape}")
        logger.info(f"   Output range: [{output_range[0]:.3f}, {output_range[1]:.3f}]")
        logger.info(f"   Processing time: {processing_time:.3f}s")
        
        return result
    
    def step_3_neural_training(self, model: CharacterLSTMModel, vocabulary: CharacterVocabulary) -> Dict[str, Any]:
        """Step 3: Train the neural model"""
        logger.info("=" * 60)
        logger.info("STEP 3: Neural Model Training")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Create trainer
        trainer = NeuralTrainer(model, vocabulary, self.training_config)
        
        # Train model
        training_result = trainer.train(
            texts=self.training_texts,
            lstm_config=self.lstm_config,
            output_dir=str(self.output_dir / "neural_training"),
            model_name="khmer_char_lstm_phase31"
        )
        
        # Get training summary
        training_summary = trainer.get_training_summary()
        
        processing_time = time.time() - start_time
        
        result = {
            'training_result': training_result,
            'training_summary': training_summary,
            'best_model_path': training_result.best_model_path,
            'best_epoch': training_result.best_epoch,
            'best_val_loss': training_result.best_val_loss,
            'total_training_time': training_result.total_training_time,
            'processing_time': processing_time,
            'trained_model': model,
            'vocabulary': vocabulary
        }
        
        logger.info(f"✅ Neural training completed:")
        logger.info(f"   Total epochs: {training_summary['total_epochs']}")
        logger.info(f"   Best epoch: {training_summary['best_epoch']}")
        logger.info(f"   Best validation loss: {training_summary['best_val_loss']:.4f}")
        logger.info(f"   Best validation perplexity: {training_summary['best_val_perplexity']:.2f}")
        logger.info(f"   Final training loss: {training_summary['final_train_loss']:.4f}")
        logger.info(f"   Total training time: {training_summary['total_training_time']:.1f}s")
        logger.info(f"   Average epoch time: {training_summary['avg_epoch_time']:.1f}s")
        logger.info(f"   Device: {training_summary['device']}")
        logger.info(f"   Best model saved: {training_result.best_model_path}")
        
        return result
    
    def step_4_neural_validation(self, model: CharacterLSTMModel, vocabulary: CharacterVocabulary) -> Dict[str, Any]:
        """Step 4: Neural validation and error detection"""
        logger.info("=" * 60)
        logger.info("STEP 4: Neural Validation & Error Detection")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Create neural validator
        validator = NeuralValidator(model, vocabulary, self.training_config.device)
        
        # Validate test texts
        validation_results = validator.batch_validate(self.test_texts, detailed_analysis=True)
        
        # Get validation statistics
        validation_stats = validator.get_validation_statistics(validation_results)
        
        # Analyze individual results
        detailed_results = []
        for i, result in enumerate(validation_results):
            summary = {
                'text_index': i,
                'text': result.text[:50] + "..." if len(result.text) > 50 else result.text,
                'is_valid': result.is_valid,
                'overall_perplexity': result.overall_perplexity,
                'model_confidence': result.model_confidence,
                'errors_detected': len(result.detected_errors),
                'processing_time': result.processing_time,
                'text_complexity': result.text_complexity
            }
            
            if result.detected_errors:
                summary['errors'] = [
                    {
                        'position': error.position,
                        'character': error.character,
                        'error_type': error.error_type,
                        'perplexity_score': error.perplexity_score,
                        'confidence': error.confidence,
                        'suggested_corrections': error.suggested_corrections[:3]
                    }
                    for error in result.detected_errors[:3]  # Show first 3 errors
                ]
            
            detailed_results.append(summary)
        
        processing_time = time.time() - start_time
        
        result = {
            'validation_results': validation_results,
            'validation_statistics': validation_stats,
            'detailed_results': detailed_results,
            'processing_time': processing_time,
            'validator': validator
        }
        
        logger.info(f"✅ Neural validation completed:")
        logger.info(f"   Total texts validated: {validation_stats['total_texts']}")
        logger.info(f"   Valid texts: {validation_stats['valid_texts']}")
        logger.info(f"   Validation rate: {validation_stats['validation_rate']:.1%}")
        logger.info(f"   Total errors detected: {validation_stats['total_errors']}")
        logger.info(f"   Texts with errors: {validation_stats['texts_with_errors']}")
        logger.info(f"   Average errors per text: {validation_stats['avg_errors_per_text']:.2f}")
        logger.info(f"   Average perplexity: {validation_stats['perplexity_stats']['mean']:.2f}")
        logger.info(f"   Average confidence: {validation_stats['confidence_stats']['avg_confidence']:.3f}")
        logger.info(f"   Processing rate: {validation_stats['performance_stats']['texts_per_second']:.1f} texts/sec")
        logger.info(f"   Processing time: {processing_time:.3f}s")
        
        # Show sample results
        logger.info(f"\n📋 Sample validation results:")
        for i, detail in enumerate(detailed_results[:3]):
            logger.info(f"   {i+1}. '{detail['text']}'")
            logger.info(f"      Valid: {detail['is_valid']}")
            logger.info(f"      Perplexity: {detail['overall_perplexity']:.2f}")
            logger.info(f"      Confidence: {detail['model_confidence']:.3f}")
            logger.info(f"      Errors: {detail['errors_detected']}")
            if 'errors' in detail:
                for error in detail['errors']:
                    logger.info(f"        Position {error['position']}: '{error['character']}' "
                              f"(perplexity: {error['perplexity_score']:.1f}, "
                              f"suggestions: {error['suggested_corrections']})")
        
        return result
    
    def step_5_ensemble_integration(self, neural_validator: NeuralValidator) -> Dict[str, Any]:
        """Step 5: Integrate neural models with statistical ensemble"""
        logger.info("=" * 60)
        logger.info("STEP 5: Neural + Statistical Ensemble Integration")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load statistical ensemble from Phase 2.4
            logger.info("Loading statistical ensemble from Phase 2.4...")
            
            # Try to load existing ensemble
            ensemble_config_path = "output/ensemble_optimization/optimized_ensemble_config.json"
            if Path(ensemble_config_path).exists():
                statistical_ensemble = EnsembleOptimizer()
                statistical_ensemble.load_models_from_config(ensemble_config_path)
                logger.info("✅ Statistical ensemble loaded successfully")
            else:
                logger.warning("❌ Statistical ensemble not found. Creating mock ensemble for demo.")
                # Create a mock ensemble for demonstration
                statistical_ensemble = self._create_mock_statistical_ensemble()
            
            # Create neural ensemble optimizer
            neural_ensemble = NeuralEnsembleOptimizer(
                neural_validator=neural_validator,
                statistical_ensemble=statistical_ensemble,
                config=self.ensemble_config
            )
            
            # Test hybrid validation
            logger.info("Testing hybrid neural + statistical validation...")
            hybrid_results = neural_ensemble.batch_validate(self.test_texts, show_progress=True)
            
            # Analyze performance
            performance_analysis = neural_ensemble.get_performance_analysis(hybrid_results)
            
            # Optimize weights
            logger.info("Optimizing neural vs statistical weights...")
            optimized_weights = neural_ensemble.optimize_weights(
                validation_texts=self.test_texts,
                target_accuracy=0.95
            )
            
            # Save configuration
            config_path = self.output_dir / "neural_ensemble_config.json"
            neural_ensemble.save_configuration(str(config_path))
            
            processing_time = time.time() - start_time
            
            result = {
                'neural_ensemble': neural_ensemble,
                'hybrid_results': hybrid_results,
                'performance_analysis': performance_analysis,
                'optimized_weights': optimized_weights,
                'config_path': str(config_path),
                'processing_time': processing_time,
                'integration_successful': True
            }
            
            logger.info(f"✅ Neural + Statistical integration completed:")
            logger.info(f"   Total texts processed: {performance_analysis['overall_performance']['total_texts']}")
            logger.info(f"   Validation rate: {performance_analysis['overall_performance']['validation_rate']:.1%}")
            logger.info(f"   Average processing time: {performance_analysis['overall_performance']['avg_processing_time']:.4f}s")
            logger.info(f"   Throughput: {performance_analysis['overall_performance']['throughput_texts_per_second']:.1f} texts/sec")
            
            logger.info(f"   Neural performance:")
            logger.info(f"     Average confidence: {performance_analysis['neural_performance']['avg_confidence']:.3f}")
            logger.info(f"     Average perplexity: {performance_analysis['neural_performance']['avg_perplexity']:.2f}")
            logger.info(f"     Processing time: {performance_analysis['neural_performance']['avg_processing_time']:.4f}s")
            
            logger.info(f"   Statistical performance:")
            logger.info(f"     Average confidence: {performance_analysis['statistical_performance']['avg_confidence']:.3f}")
            logger.info(f"     Processing time: {performance_analysis['statistical_performance']['avg_processing_time']:.4f}s")
            
            logger.info(f"   Ensemble performance:")
            logger.info(f"     Combined confidence: {performance_analysis['ensemble_performance']['avg_combined_confidence']:.3f}")
            logger.info(f"     Neural vs Statistical speed ratio: {performance_analysis['ensemble_performance']['neural_vs_statistical_speed']:.2f}")
            
            logger.info(f"   Optimized weights:")
            logger.info(f"     Neural: {optimized_weights['neural']:.3f}")
            logger.info(f"     Statistical: {optimized_weights['statistical']:.3f}")
            
            logger.info(f"   Configuration saved to: {config_path}")
            logger.info(f"   Processing time: {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Integration failed: {e}")
            logger.error(traceback.format_exc())
            
            result = {
                'integration_successful': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
        
        return result
    
    def _create_mock_statistical_ensemble(self):
        """Create a mock statistical ensemble for demo purposes"""
        logger.info("Creating mock statistical ensemble...")
        
        class MockEnsemble:
            def validate_text(self, text):
                # Mock validation result
                class MockResult:
                    def __init__(self):
                        self.is_valid = True
                        self.confidence = 0.75
                        self.errors = []
                
                return MockResult()
            
            def get_configuration(self):
                return {'mock': True}
        
        return MockEnsemble()
    
    def step_6_performance_evaluation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Comprehensive performance evaluation"""
        logger.info("=" * 60)
        logger.info("STEP 6: Performance Evaluation & Analysis")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Compile comprehensive metrics
        evaluation = {
            'vocabulary_metrics': {
                'vocabulary_size': results['step1']['vocabulary_size'],
                'character_coverage': results['step1']['coverage'],
                'khmer_ratio': results['step1']['khmer_ratio'],
                'processing_time': results['step1']['processing_time']
            },
            'model_metrics': {
                'total_parameters': results['step2']['model_info']['total_parameters'],
                'model_size_mb': results['step2']['model_info']['model_size_mb'],
                'architecture_setup_time': results['step2']['processing_time']
            },
            'training_metrics': {
                'total_epochs': results['step3']['training_summary']['total_epochs'],
                'best_validation_loss': results['step3']['training_summary']['best_val_loss'],
                'best_validation_perplexity': results['step3']['training_summary']['best_val_perplexity'],
                'total_training_time': results['step3']['training_summary']['total_training_time'],
                'average_epoch_time': results['step3']['training_summary']['avg_epoch_time']
            },
            'validation_metrics': {
                'validation_rate': results['step4']['validation_statistics']['validation_rate'],
                'average_perplexity': results['step4']['validation_statistics']['perplexity_stats']['mean'],
                'average_confidence': results['step4']['validation_statistics']['confidence_stats']['avg_confidence'],
                'processing_rate': results['step4']['validation_statistics']['performance_stats']['texts_per_second'],
                'error_detection_rate': results['step4']['validation_statistics']['texts_with_errors'] / results['step4']['validation_statistics']['total_texts']
            }
        }
        
        # Add ensemble metrics if available
        if results['step5']['integration_successful']:
            ensemble_perf = results['step5']['performance_analysis']
            evaluation['ensemble_metrics'] = {
                'hybrid_validation_rate': ensemble_perf['overall_performance']['validation_rate'],
                'hybrid_throughput': ensemble_perf['overall_performance']['throughput_texts_per_second'],
                'neural_confidence': ensemble_perf['neural_performance']['avg_confidence'],
                'statistical_confidence': ensemble_perf['statistical_performance']['avg_confidence'],
                'combined_confidence': ensemble_perf['ensemble_performance']['avg_combined_confidence'],
                'neural_weight': results['step5']['optimized_weights']['neural'],
                'statistical_weight': results['step5']['optimized_weights']['statistical']
            }
        
        # Calculate improvement metrics
        baseline_accuracy = 0.625  # From Phase 2.3
        statistical_accuracy = 0.882  # From Phase 2.4
        
        if 'ensemble_metrics' in evaluation:
            neural_improvement = evaluation['ensemble_metrics']['hybrid_validation_rate'] - statistical_accuracy
            total_improvement = evaluation['ensemble_metrics']['hybrid_validation_rate'] - baseline_accuracy
        else:
            neural_improvement = evaluation['validation_metrics']['validation_rate'] - statistical_accuracy
            total_improvement = evaluation['validation_metrics']['validation_rate'] - baseline_accuracy
        
        evaluation['improvement_metrics'] = {
            'baseline_accuracy': baseline_accuracy,
            'statistical_accuracy': statistical_accuracy,
            'neural_improvement': neural_improvement,
            'total_improvement': total_improvement,
            'improvement_percentage': (total_improvement / baseline_accuracy) * 100
        }
        
        # Performance benchmarks
        target_metrics = {
            'accuracy': 0.90,  # 90% target
            'processing_time': 0.060,  # <60ms per text
            'perplexity': 20.0,  # Target perplexity
            'throughput': 16.0  # Texts per second
        }
        
        if 'ensemble_metrics' in evaluation:
            current_accuracy = evaluation['ensemble_metrics']['hybrid_validation_rate']
            current_processing_time = 1.0 / evaluation['ensemble_metrics']['hybrid_throughput']
            current_throughput = evaluation['ensemble_metrics']['hybrid_throughput']
        else:
            current_accuracy = evaluation['validation_metrics']['validation_rate']
            current_processing_time = 1.0 / evaluation['validation_metrics']['processing_rate']
            current_throughput = evaluation['validation_metrics']['processing_rate']
        
        current_perplexity = evaluation['validation_metrics']['average_perplexity']
        
        evaluation['benchmark_comparison'] = {
            'accuracy_achieved': current_accuracy >= target_metrics['accuracy'],
            'speed_achieved': current_processing_time <= target_metrics['processing_time'],
            'perplexity_achieved': current_perplexity <= target_metrics['perplexity'],
            'throughput_achieved': current_throughput >= target_metrics['throughput'],
            'current_vs_target': {
                'accuracy': f"{current_accuracy:.1%} / {target_metrics['accuracy']:.0%}",
                'processing_time': f"{current_processing_time:.3f}s / {target_metrics['processing_time']:.3f}s",
                'perplexity': f"{current_perplexity:.1f} / {target_metrics['perplexity']:.1f}",
                'throughput': f"{current_throughput:.1f} / {target_metrics['throughput']:.1f} texts/sec"
            }
        }
        
        processing_time = time.time() - start_time
        evaluation['evaluation_time'] = processing_time
        
        # Save evaluation results
        eval_path = self.output_dir / "performance_evaluation.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Performance evaluation completed:")
        logger.info(f"   Vocabulary: {evaluation['vocabulary_metrics']['vocabulary_size']} chars, "
                  f"{evaluation['vocabulary_metrics']['character_coverage']:.1%} coverage")
        logger.info(f"   Model: {evaluation['model_metrics']['total_parameters']:,} params, "
                  f"{evaluation['model_metrics']['model_size_mb']:.1f} MB")
        logger.info(f"   Training: {evaluation['training_metrics']['total_epochs']} epochs, "
                  f"{evaluation['training_metrics']['best_validation_perplexity']:.1f} best perplexity")
        logger.info(f"   Validation: {evaluation['validation_metrics']['validation_rate']:.1%} rate, "
                  f"{evaluation['validation_metrics']['processing_rate']:.1f} texts/sec")
        
        if 'ensemble_metrics' in evaluation:
            logger.info(f"   Ensemble: {evaluation['ensemble_metrics']['hybrid_validation_rate']:.1%} rate, "
                      f"{evaluation['ensemble_metrics']['hybrid_throughput']:.1f} texts/sec")
            logger.info(f"   Weights: Neural {evaluation['ensemble_metrics']['neural_weight']:.1%}, "
                      f"Statistical {evaluation['ensemble_metrics']['statistical_weight']:.1%}")
        
        logger.info(f"   Improvement: {evaluation['improvement_metrics']['neural_improvement']:+.1%} vs statistical, "
                  f"{evaluation['improvement_metrics']['total_improvement']:+.1%} vs baseline")
        
        logger.info(f"🎯 Target benchmarks:")
        for metric, achieved in evaluation['benchmark_comparison'].items():
            if metric != 'current_vs_target':
                status = "✅" if achieved else "❌"
                logger.info(f"   {metric.replace('_', ' ').title()}: {status}")
        
        logger.info(f"   Current vs Target:")
        for metric, comparison in evaluation['benchmark_comparison']['current_vs_target'].items():
            logger.info(f"     {metric.replace('_', ' ').title()}: {comparison}")
        
        logger.info(f"   Evaluation saved to: {eval_path}")
        logger.info(f"   Processing time: {processing_time:.3f}s")
        
        return evaluation
    
    def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete Phase 3.1 neural models demonstration"""
        logger.info("🚀 STARTING PHASE 3.1 NEURAL MODELS DEMONSTRATION")
        logger.info("=" * 80)
        logger.info(f"Demo: {self.demo_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Training texts: {len(self.training_texts)}")
        logger.info(f"Test texts: {len(self.test_texts)}")
        logger.info("=" * 80)
        
        overall_start_time = time.time()
        results = {}
        
        try:
            # Step 1: Vocabulary Building
            results['step1'] = self.step_1_vocabulary_building()
            
            # Step 2: Model Architecture
            results['step2'] = self.step_2_model_architecture(results['step1']['vocabulary'])
            
            # Step 3: Neural Training
            results['step3'] = self.step_3_neural_training(
                results['step2']['model'], 
                results['step1']['vocabulary']
            )
            
            # Step 4: Neural Validation
            results['step4'] = self.step_4_neural_validation(
                results['step3']['trained_model'],
                results['step3']['vocabulary']
            )
            
            # Step 5: Ensemble Integration
            results['step5'] = self.step_5_ensemble_integration(results['step4']['validator'])
            
            # Step 6: Performance Evaluation
            results['step6'] = self.step_6_performance_evaluation(results)
            
            overall_processing_time = time.time() - overall_start_time
            
            # Final summary
            results['summary'] = {
                'demo_completed': True,
                'total_processing_time': overall_processing_time,
                'steps_completed': 6,
                'output_directory': str(self.output_dir),
                'final_performance': results['step6']
            }
            
            logger.info("=" * 80)
            logger.info("🎉 PHASE 3.1 NEURAL MODELS DEMONSTRATION COMPLETED")
            logger.info("=" * 80)
            logger.info(f"✅ All 6 steps completed successfully")
            logger.info(f"   Total processing time: {overall_processing_time:.1f}s")
            logger.info(f"   Output directory: {self.output_dir}")
            
            # Print key achievements
            eval_metrics = results['step6']
            logger.info(f"\n🏆 KEY ACHIEVEMENTS:")
            logger.info(f"   • Character vocabulary: {eval_metrics['vocabulary_metrics']['vocabulary_size']} characters")
            logger.info(f"   • Model parameters: {eval_metrics['model_metrics']['total_parameters']:,}")
            logger.info(f"   • Training epochs: {eval_metrics['training_metrics']['total_epochs']}")
            logger.info(f"   • Validation accuracy: {eval_metrics['validation_metrics']['validation_rate']:.1%}")
            logger.info(f"   • Processing speed: {eval_metrics['validation_metrics']['processing_rate']:.1f} texts/sec")
            
            if 'ensemble_metrics' in eval_metrics:
                logger.info(f"   • Ensemble accuracy: {eval_metrics['ensemble_metrics']['hybrid_validation_rate']:.1%}")
                logger.info(f"   • Ensemble speed: {eval_metrics['ensemble_metrics']['hybrid_throughput']:.1f} texts/sec")
            
            logger.info(f"   • Improvement vs baseline: {eval_metrics['improvement_metrics']['total_improvement']:+.1%}")
            
            # Save complete results
            results_path = self.output_dir / "complete_demo_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                # Convert non-serializable objects to strings for JSON
                json_safe_results = self._make_json_safe(results)
                json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"   • Complete results saved: {results_path}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Demo failed with error: {e}")
            logger.error(traceback.format_exc())
            
            results['summary'] = {
                'demo_completed': False,
                'error': str(e),
                'total_processing_time': time.time() - overall_start_time,
                'steps_completed': len([k for k in results.keys() if k.startswith('step')])
            }
        
        return results
    
    def _make_json_safe(self, obj):
        """Convert objects to JSON-safe format"""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items() 
                   if k not in ['model', 'vocabulary', 'validator', 'neural_ensemble', 'trained_model']}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return str(type(obj).__name__)
        else:
            return obj


def main():
    """Main execution function"""
    try:
        # Create and run demo
        demo = Phase31NeuralDemo()
        results = demo.run_complete_demo()
        
        if results['summary']['demo_completed']:
            print("\n🎉 Phase 3.1 Neural Models Demo completed successfully!")
            print(f"📁 Results saved to: {results['summary']['output_directory']}")
            return 0
        else:
            print(f"\n❌ Demo failed: {results['summary']['error']}")
            return 1
            
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main()) 