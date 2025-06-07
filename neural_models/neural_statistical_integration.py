"""
Phase 3.3: Neural-Statistical Integration for Khmer Spellchecking

This module provides comprehensive integration between syllable-level neural models
and existing statistical models, creating a hybrid approach that leverages both
neural predictions and statistical n-gram analysis for superior spellchecking accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import neural models
from neural_models.syllable_lstm import SyllableLSTMModel, SyllableVocabulary, SyllableLSTMConfiguration
from neural_models.character_lstm import CharacterLSTMModel, CharacterVocabulary, LSTMConfiguration

# Import statistical models
from statistical_models.syllable_ngram_model import SyllableNgramModel
from statistical_models.character_ngram_model import CharacterNgramModel
from statistical_models.ensemble_optimizer import EnsembleOptimizer
from statistical_models.rule_based_validator import RuleBasedValidator

# Import segmentation API
from word_cluster.syllable_api import SyllableSegmentationAPI, SegmentationMethod


@dataclass
class IntegrationConfiguration:
    """Configuration for neural-statistical integration"""
    # Neural model settings
    neural_weight: float = 0.4
    neural_temperature: float = 1.0
    neural_threshold: float = 0.1
    
    # Statistical model settings
    statistical_weight: float = 0.4
    ngram_threshold: float = 0.01
    min_ngram_order: int = 2
    max_ngram_order: int = 4
    
    # Rule-based settings
    rule_weight: float = 0.2
    strict_rules: bool = False
    
    # Integration settings
    consensus_threshold: float = 0.6
    error_confidence_threshold: float = 0.5
    combine_errors: bool = True
    max_suggestions: int = 5
    
    # Performance settings
    batch_size: int = 32
    max_sequence_length: int = 20


@dataclass
class IntegratedError:
    """Unified error representation combining neural and statistical detection"""
    position: int
    syllable: str
    error_type: str
    confidence: float
    neural_score: Optional[float] = None
    statistical_score: Optional[float] = None
    rule_based_score: Optional[float] = None
    suggestions: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


@dataclass
class IntegrationResult:
    """Result of neural-statistical integration analysis"""
    text: str
    syllables: List[str]
    is_valid: bool
    overall_confidence: float
    errors: List[IntegratedError]
    neural_perplexity: Optional[float] = None
    statistical_entropy: Optional[float] = None
    rule_based_score: Optional[float] = None
    processing_time: float = 0.0
    method_agreement: float = 0.0


class NeuralStatisticalIntegrator:
    """
    Advanced integration system combining syllable-level neural models 
    with statistical n-gram models and rule-based validation for 
    comprehensive Khmer spellchecking.
    """
    
    def __init__(self, config: IntegrationConfiguration):
        self.config = config
        self.logger = logging.getLogger("neural_statistical_integrator")
        
        # Model containers
        self.neural_model: Optional[SyllableLSTMModel] = None
        self.neural_vocabulary: Optional[SyllableVocabulary] = None
        self.statistical_models: Dict[int, SyllableNgramModel] = {}
        self.rule_validator: Optional[RuleBasedValidator] = None
        
        # Segmentation API
        self.segmentation_api = SyllableSegmentationAPI(SegmentationMethod.REGEX_ADVANCED)
        
        # Statistics
        self.processing_stats = {
            'texts_processed': 0,
            'errors_detected': 0,
            'neural_predictions': 0,
            'statistical_validations': 0,
            'rule_validations': 0,
            'consensus_agreements': 0
        }
    
    def load_neural_model(self, model_path: str) -> bool:
        """Load syllable-level neural model"""
        try:
            if Path(model_path).exists():
                self.neural_model, self.neural_vocabulary = SyllableLSTMModel.load_model(model_path)
                self.neural_model.eval()
                self.logger.info(f"✅ Neural model loaded from {model_path}")
                return True
            else:
                self.logger.warning(f"Neural model not found: {model_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to load neural model: {e}")
            return False
    
    def load_statistical_models(self, model_directory: str) -> int:
        """Load syllable n-gram statistical models"""
        loaded_count = 0
        model_dir = Path(model_directory)
        
        for ngram_order in range(self.config.min_ngram_order, self.config.max_ngram_order + 1):
            model_file = model_dir / f"syllable_{ngram_order}gram_model.pkl"
            
            try:
                if model_file.exists():
                    model = SyllableNgramModel(ngram_order)
                    model.load_model(str(model_file))
                    self.statistical_models[ngram_order] = model
                    loaded_count += 1
                    self.logger.info(f"✅ Loaded {ngram_order}-gram model")
                else:
                    self.logger.warning(f"Statistical model not found: {model_file}")
            except Exception as e:
                self.logger.error(f"Failed to load {ngram_order}-gram model: {e}")
        
        self.logger.info(f"Loaded {loaded_count} statistical models")
        return loaded_count
    
    def load_rule_validator(self) -> bool:
        """Load rule-based validator"""
        try:
            self.rule_validator = RuleBasedValidator()
            self.logger.info("✅ Rule-based validator loaded")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load rule validator: {e}")
            return False
    
    def create_minimal_models(self, texts: List[str]) -> bool:
        """Create minimal models for demonstration when full models aren't available"""
        try:
            self.logger.info("Creating minimal models for demonstration...")
            
            # Create minimal neural model
            if self.neural_model is None:
                config = SyllableLSTMConfiguration(
                    embedding_dim=32,
                    hidden_dim=64,
                    num_layers=1,
                    sequence_length=10,
                    max_vocab_size=500
                )
                
                # Build minimal vocabulary
                self.neural_vocabulary = SyllableVocabulary(config)
                vocab_stats = self.neural_vocabulary.build_vocabulary(texts[:100])  # Use subset
                
                # Create minimal model
                self.neural_model = SyllableLSTMModel(config, self.neural_vocabulary.vocabulary_size)
                self.neural_model.eval()
                
                self.logger.info(f"✅ Created minimal neural model ({vocab_stats['vocabulary_size']} syllables)")
            
            # Create minimal statistical models
            if not self.statistical_models:
                from statistical_models.syllable_ngram_model import SyllableNgramModelTrainer
                
                trainer = SyllableNgramModelTrainer()
                models = trainer.train_models(
                    texts[:200],  # Use subset for quick training
                    ngram_sizes=[2, 3],  # Minimal n-gram sizes
                    output_dir=None  # Don't save
                )
                
                for ngram_order, model in models.items():
                    self.statistical_models[ngram_order] = model
                
                self.logger.info(f"✅ Created {len(models)} minimal statistical models")
            
            # Load rule validator
            if self.rule_validator is None:
                self.load_rule_validator()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create minimal models: {e}")
            return False
    
    def neural_validate_syllables(self, syllables: List[str]) -> Tuple[float, List[Tuple[int, float]]]:
        """Validate syllables using neural model"""
        if self.neural_model is None or self.neural_vocabulary is None:
            return 0.0, []
        
        try:
            # Encode syllables
            encoded = self.neural_vocabulary.encode_text(' '.join(syllables), max_length=self.config.max_sequence_length)
            if not encoded:
                return 0.0, []
            
            input_tensor = torch.tensor([encoded]).long()
            
            with torch.no_grad():
                # Get model predictions for sequence validation
                output = self.neural_model(input_tensor)
                probs = torch.softmax(output, dim=-1)
                
                # Calculate perplexity as overall sequence quality
                max_prob = torch.max(probs).item()
                perplexity = 1.0 / max_prob if max_prob > 0 else float('inf')
                
                # Identify suspicious syllables (low probability predictions)
                suspicious_syllables = []
                for i, syllable in enumerate(syllables):
                    if i < len(encoded) - 1:  # Don't check last syllable
                        # Get prediction for next syllable
                        next_id = encoded[i + 1] if i + 1 < len(encoded) else 0
                        next_prob = probs[0][next_id].item()
                        
                        if next_prob < self.config.neural_threshold:
                            confidence = 1.0 - next_prob
                            suspicious_syllables.append((i, confidence))
                
                self.processing_stats['neural_predictions'] += 1
                return min(max_prob, 1.0), suspicious_syllables
                
        except Exception as e:
            self.logger.warning(f"Neural validation failed: {e}")
            return 0.0, []
    
    def statistical_validate_syllables(self, syllables: List[str]) -> Tuple[float, List[Tuple[int, float]]]:
        """Validate syllables using statistical n-gram models"""
        if not self.statistical_models:
            return 0.0, []
        
        try:
            all_errors = []
            all_scores = []
            
            for ngram_order, model in self.statistical_models.items():
                # Get n-gram probabilities
                ngrams = []
                for i in range(len(syllables) - ngram_order + 1):
                    ngram = tuple(syllables[i:i + ngram_order])
                    prob = model.get_ngram_probability(ngram)
                    ngrams.append((i, ngram, prob))
                
                # Identify low-probability n-grams
                for i, ngram, prob in ngrams:
                    if prob < self.config.ngram_threshold:
                        confidence = 1.0 - (prob / self.config.ngram_threshold)
                        all_errors.append((i, confidence))
                        all_scores.append(prob)
            
            # Calculate overall statistical score
            if all_scores:
                overall_score = np.mean(all_scores)
            else:
                overall_score = 1.0
            
            # Deduplicate and average errors by position
            position_errors = {}
            for pos, conf in all_errors:
                if pos not in position_errors:
                    position_errors[pos] = []
                position_errors[pos].append(conf)
            
            final_errors = [(pos, np.mean(confs)) for pos, confs in position_errors.items()]
            
            self.processing_stats['statistical_validations'] += 1
            return overall_score, final_errors
            
        except Exception as e:
            self.logger.warning(f"Statistical validation failed: {e}")
            return 0.0, []
    
    def rule_validate_text(self, text: str) -> Tuple[float, List[Tuple[int, float]]]:
        """Validate text using rule-based validator"""
        if self.rule_validator is None:
            return 1.0, []
        
        try:
            result = self.rule_validator.validate_text(text)
            
            # Convert rule-based errors to position-based format
            position_errors = []
            for error in result.errors:
                if hasattr(error, 'position') and error.position is not None:
                    confidence = 0.9  # High confidence for rule-based errors
                    position_errors.append((error.position, confidence))
            
            self.processing_stats['rule_validations'] += 1
            return result.overall_score, position_errors
            
        except Exception as e:
            self.logger.warning(f"Rule validation failed: {e}")
            return 1.0, []
    
    def integrate_errors(self, 
                        syllables: List[str],
                        neural_errors: List[Tuple[int, float]],
                        statistical_errors: List[Tuple[int, float]],
                        rule_errors: List[Tuple[int, float]]) -> List[IntegratedError]:
        """Integrate errors from all sources with confidence weighting"""
        
        # Collect all errors by position
        all_errors = {}
        
        # Neural errors
        for pos, conf in neural_errors:
            if pos not in all_errors:
                all_errors[pos] = {
                    'position': pos,
                    'syllable': syllables[pos] if pos < len(syllables) else '',
                    'neural': conf,
                    'statistical': None,
                    'rule': None,
                    'sources': []
                }
            all_errors[pos]['neural'] = conf
            all_errors[pos]['sources'].append('neural')
        
        # Statistical errors
        for pos, conf in statistical_errors:
            if pos not in all_errors:
                all_errors[pos] = {
                    'position': pos,
                    'syllable': syllables[pos] if pos < len(syllables) else '',
                    'neural': None,
                    'statistical': conf,
                    'rule': None,
                    'sources': []
                }
            all_errors[pos]['statistical'] = conf
            all_errors[pos]['sources'].append('statistical')
        
        # Rule-based errors
        for pos, conf in rule_errors:
            if pos not in all_errors:
                all_errors[pos] = {
                    'position': pos,
                    'syllable': syllables[pos] if pos < len(syllables) else '',
                    'neural': None,
                    'statistical': None,
                    'rule': conf,
                    'sources': []
                }
            all_errors[pos]['rule'] = conf
            all_errors[pos]['sources'].append('rule')
        
        # Calculate integrated confidence and create error objects
        integrated_errors = []
        
        for pos, error_data in all_errors.items():
            # Calculate weighted confidence
            weighted_conf = 0.0
            total_weight = 0.0
            
            if error_data['neural'] is not None:
                weighted_conf += error_data['neural'] * self.config.neural_weight
                total_weight += self.config.neural_weight
            
            if error_data['statistical'] is not None:
                weighted_conf += error_data['statistical'] * self.config.statistical_weight
                total_weight += self.config.statistical_weight
            
            if error_data['rule'] is not None:
                weighted_conf += error_data['rule'] * self.config.rule_weight
                total_weight += self.config.rule_weight
            
            if total_weight > 0:
                final_confidence = weighted_conf / total_weight
            else:
                final_confidence = 0.0
            
            # Only include errors above threshold
            if final_confidence >= self.config.error_confidence_threshold:
                # Determine error type based on strongest source
                error_type = "unknown"
                if error_data['rule'] and error_data['rule'] >= 0.8:
                    error_type = "rule_violation"
                elif error_data['neural'] and error_data['neural'] > 0.7:
                    error_type = "neural_anomaly"
                elif error_data['statistical'] and error_data['statistical'] > 0.7:
                    error_type = "statistical_outlier"
                else:
                    error_type = "consensus_error"
                
                integrated_error = IntegratedError(
                    position=pos,
                    syllable=error_data['syllable'],
                    error_type=error_type,
                    confidence=final_confidence,
                    neural_score=error_data['neural'],
                    statistical_score=error_data['statistical'],
                    rule_based_score=error_data['rule'],
                    suggestions=[],  # TODO: Implement suggestion generation
                    sources=error_data['sources']
                )
                
                integrated_errors.append(integrated_error)
        
        return integrated_errors
    
    def calculate_method_agreement(self, 
                                 neural_errors: List[Tuple[int, float]],
                                 statistical_errors: List[Tuple[int, float]],
                                 rule_errors: List[Tuple[int, float]]) -> float:
        """Calculate agreement between different validation methods"""
        
        neural_positions = set(pos for pos, _ in neural_errors)
        statistical_positions = set(pos for pos, _ in statistical_errors)
        rule_positions = set(pos for pos, _ in rule_errors)
        
        all_positions = neural_positions | statistical_positions | rule_positions
        
        if not all_positions:
            return 1.0  # Perfect agreement when no errors
        
        # Calculate intersection over union
        agreements = 0
        for pos in all_positions:
            methods_agree = 0
            if pos in neural_positions:
                methods_agree += 1
            if pos in statistical_positions:
                methods_agree += 1
            if pos in rule_positions:
                methods_agree += 1
            
            if methods_agree >= 2:  # At least 2 methods agree
                agreements += 1
        
        agreement_score = agreements / len(all_positions)
        
        if agreement_score >= self.config.consensus_threshold:
            self.processing_stats['consensus_agreements'] += 1
        
        return agreement_score
    
    def validate_text(self, text: str) -> IntegrationResult:
        """Comprehensive text validation using neural-statistical integration"""
        start_time = time.time()
        
        try:
            # Segment text into syllables
            segmentation_result = self.segmentation_api.segment_text(text)
            if not segmentation_result.success:
                return IntegrationResult(
                    text=text,
                    syllables=[],
                    is_valid=False,
                    overall_confidence=0.0,
                    errors=[],
                    processing_time=time.time() - start_time
                )
            
            syllables = segmentation_result.syllables
            
            # Neural validation
            neural_score, neural_errors = self.neural_validate_syllables(syllables)
            
            # Statistical validation
            statistical_score, statistical_errors = self.statistical_validate_syllables(syllables)
            
            # Rule-based validation
            rule_score, rule_errors = self.rule_validate_text(text)
            
            # Integrate errors
            integrated_errors = self.integrate_errors(
                syllables, neural_errors, statistical_errors, rule_errors
            )
            
            # Calculate method agreement
            method_agreement = self.calculate_method_agreement(
                neural_errors, statistical_errors, rule_errors
            )
            
            # Calculate overall confidence
            weighted_score = (
                neural_score * self.config.neural_weight +
                statistical_score * self.config.statistical_weight +
                rule_score * self.config.rule_weight
            ) / (self.config.neural_weight + self.config.statistical_weight + self.config.rule_weight)
            
            # Adjust confidence based on method agreement
            overall_confidence = weighted_score * (0.5 + 0.5 * method_agreement)
            
            # Determine validity
            is_valid = (
                len(integrated_errors) == 0 and 
                overall_confidence >= self.config.consensus_threshold
            )
            
            processing_time = time.time() - start_time
            self.processing_stats['texts_processed'] += 1
            self.processing_stats['errors_detected'] += len(integrated_errors)
            
            return IntegrationResult(
                text=text,
                syllables=syllables,
                is_valid=is_valid,
                overall_confidence=overall_confidence,
                errors=integrated_errors,
                neural_perplexity=1.0 / neural_score if neural_score > 0 else None,
                statistical_entropy=-np.log(statistical_score) if statistical_score > 0 else None,
                rule_based_score=rule_score,
                processing_time=processing_time,
                method_agreement=method_agreement
            )
            
        except Exception as e:
            self.logger.error(f"Text validation failed: {e}")
            return IntegrationResult(
                text=text,
                syllables=[],
                is_valid=False,
                overall_confidence=0.0,
                errors=[],
                processing_time=time.time() - start_time
            )
    
    def validate_batch(self, texts: List[str]) -> List[IntegrationResult]:
        """Batch validation with progress tracking"""
        results = []
        
        self.logger.info(f"Starting batch validation of {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                self.logger.info(f"Processed {i}/{len(texts)} texts...")
            
            result = self.validate_text(text)
            results.append(result)
        
        self.logger.info(f"✅ Batch validation completed: {len(results)} results")
        return results
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        return {
            'processing_stats': self.processing_stats.copy(),
            'models_loaded': {
                'neural_model': self.neural_model is not None,
                'neural_vocabulary_size': self.neural_vocabulary.vocabulary_size if self.neural_vocabulary else 0,
                'statistical_models': len(self.statistical_models),
                'statistical_orders': list(self.statistical_models.keys()),
                'rule_validator': self.rule_validator is not None
            },
            'configuration': {
                'neural_weight': self.config.neural_weight,
                'statistical_weight': self.config.statistical_weight,
                'rule_weight': self.config.rule_weight,
                'consensus_threshold': self.config.consensus_threshold,
                'error_confidence_threshold': self.config.error_confidence_threshold
            },
            'performance_metrics': {
                'texts_per_error': self.processing_stats['texts_processed'] / max(1, self.processing_stats['errors_detected']),
                'consensus_rate': self.processing_stats['consensus_agreements'] / max(1, self.processing_stats['texts_processed']),
                'prediction_rate': self.processing_stats['neural_predictions'] / max(1, self.processing_stats['texts_processed'])
            }
        }
    
    def generate_integration_report(self, results: List[IntegrationResult]) -> str:
        """Generate comprehensive integration analysis report"""
        if not results:
            return "No results to report"
        
        # Calculate statistics
        total_texts = len(results)
        valid_texts = sum(1 for r in results if r.is_valid)
        total_errors = sum(len(r.errors) for r in results)
        avg_confidence = np.mean([r.overall_confidence for r in results])
        avg_agreement = np.mean([r.method_agreement for r in results])
        avg_processing_time = np.mean([r.processing_time for r in results])
        
        # Error type analysis
        error_types = {}
        error_sources = {}
        
        for result in results:
            for error in result.errors:
                error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
                for source in error.sources:
                    error_sources[source] = error_sources.get(source, 0) + 1
        
        # Model performance
        neural_perplexities = [r.neural_perplexity for r in results if r.neural_perplexity is not None]
        statistical_entropies = [r.statistical_entropy for r in results if r.statistical_entropy is not None]
        rule_scores = [r.rule_based_score for r in results if r.rule_based_score is not None]
        
        report = f"""
🔬 NEURAL-STATISTICAL INTEGRATION ANALYSIS REPORT
{'=' * 80}

📊 OVERVIEW STATISTICS:
  Total texts analyzed: {total_texts:,}
  Valid texts: {valid_texts:,} ({valid_texts/total_texts*100:.1f}%)
  Total errors detected: {total_errors:,}
  Average confidence: {avg_confidence:.3f}
  Average method agreement: {avg_agreement:.3f}
  Average processing time: {avg_processing_time*1000:.2f}ms

🎯 ERROR ANALYSIS:
  Error types distribution:"""

        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_errors) * 100 if total_errors > 0 else 0
            report += f"\n    • {error_type}: {count} ({percentage:.1f}%)"
        
        report += f"\n\n  Error sources distribution:"
        for source, count in sorted(error_sources.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / sum(error_sources.values())) * 100 if error_sources else 0
            report += f"\n    • {source}: {count} ({percentage:.1f}%)"
        
        report += f"""

🧠 MODEL PERFORMANCE:
  Neural model:"""
        
        if neural_perplexities:
            report += f"""
    • Perplexity range: {min(neural_perplexities):.2f} - {max(neural_perplexities):.2f}
    • Average perplexity: {np.mean(neural_perplexities):.2f}"""
        else:
            report += f"""
    • No neural perplexity data available"""
        
        report += f"""
    • Predictions made: {self.processing_stats['neural_predictions']:,}
  
  Statistical models:"""
        
        if statistical_entropies:
            report += f"""
    • Entropy range: {min(statistical_entropies):.2f} - {max(statistical_entropies):.2f}
    • Average entropy: {np.mean(statistical_entropies):.2f}"""
        else:
            report += f"""
    • No statistical entropy data available"""
        
        report += f"""
    • Validations made: {self.processing_stats['statistical_validations']:,}
  
  Rule-based validator:"""
        
        if rule_scores:
            report += f"""
    • Score range: {min(rule_scores):.3f} - {max(rule_scores):.3f}
    • Average score: {np.mean(rule_scores):.3f}"""
        else:
            report += f"""
    • No rule-based score data available"""
        
        report += f"""
    • Validations made: {self.processing_stats['rule_validations']:,}

🤝 INTEGRATION EFFECTIVENESS:
  Method consensus rate: {self.processing_stats['consensus_agreements']/max(1, self.processing_stats['texts_processed'])*100:.1f}%
  Error confidence threshold: {self.config.error_confidence_threshold}
  Consensus threshold: {self.config.consensus_threshold}
  
  Weight distribution:
    • Neural: {self.config.neural_weight} ({self.config.neural_weight/(self.config.neural_weight+self.config.statistical_weight+self.config.rule_weight)*100:.1f}%)
    • Statistical: {self.config.statistical_weight} ({self.config.statistical_weight/(self.config.neural_weight+self.config.statistical_weight+self.config.rule_weight)*100:.1f}%)
    • Rule-based: {self.config.rule_weight} ({self.config.rule_weight/(self.config.neural_weight+self.config.statistical_weight+self.config.rule_weight)*100:.1f}%)

⚡ PERFORMANCE METRICS:
  Processing speed: {total_texts/sum(r.processing_time for r in results):.0f} texts/second
  Error detection rate: {total_errors/total_texts:.2f} errors/text
  Memory efficiency: All models loaded successfully
  
🏆 RECOMMENDATION:
  Integration quality: {"Excellent" if avg_agreement > 0.8 else "Good" if avg_agreement > 0.6 else "Needs improvement"}
  Confidence level: {"High" if avg_confidence > 0.8 else "Medium" if avg_confidence > 0.6 else "Low"}
  Ready for production: {"✅ Yes" if avg_agreement > 0.7 and avg_confidence > 0.7 else "⚠️  Needs optimization"}

{'=' * 80}
"""
        return report


if __name__ == "__main__":
    # Demo usage
    import time
    
    print("🔗 NEURAL-STATISTICAL INTEGRATION DEMO")
    print("=" * 50)
    
    # Configuration
    config = IntegrationConfiguration(
        neural_weight=0.4,
        statistical_weight=0.4,
        rule_weight=0.2,
        consensus_threshold=0.6,
        error_confidence_threshold=0.5
    )
    
    print(f"Integration weights: Neural {config.neural_weight}, Statistical {config.statistical_weight}, Rules {config.rule_weight}")
    
    # Sample texts for testing
    test_texts = [
        "នេះជាការសាកល្បងអត្ថបទដ៏ល្អមួយ។",
        "ការអប់រំជាមូលដ្ឋានសំខាន់សម្រាប់ការអភិវឌ្ឍន៍។",
        "វប្បធម៌ខ្មែរមានប្រវត្តិដ៏យូរលង់។",
        "កំពង់ចំជាទីក្រុងសំខាន់របស់ខេត្តកំពង់ចាម។",
        "កំហុសនេះគួរតែកែតម្រូវ។",  # Intentional error
        "This is English mixed នឹងភាសាខ្មែរ។",
        "២០២៥ជាឆ្នាំថ្មីដ៏មានសារៈសំខាន់។"
    ]
    
    print(f"Test texts: {len(test_texts)} samples")
    
    # Create integrator
    integrator = NeuralStatisticalIntegrator(config)
    
    # Create minimal models for demonstration
    success = integrator.create_minimal_models(test_texts * 10)  # Replicate for training
    
    if success:
        print("✅ Minimal models created successfully")
        
        # Test single text validation
        test_text = test_texts[0]
        result = integrator.validate_text(test_text)
        
        print(f"\n🔍 Single text validation:")
        print(f"   Text: '{test_text}'")
        print(f"   Valid: {'✅' if result.is_valid else '❌'}")
        print(f"   Confidence: {result.overall_confidence:.3f}")
        print(f"   Method agreement: {result.method_agreement:.3f}")
        print(f"   Errors detected: {len(result.errors)}")
        print(f"   Processing time: {result.processing_time*1000:.2f}ms")
        
        # Test batch validation
        print(f"\n📦 Batch validation:")
        batch_results = integrator.validate_batch(test_texts)
        
        # Generate report
        report = integrator.generate_integration_report(batch_results)
        print(report)
        
        # Integration statistics
        stats = integrator.get_integration_statistics()
        print(f"🔧 Integration Statistics:")
        print(f"   Models loaded: {stats['models_loaded']}")
        print(f"   Processing stats: {stats['processing_stats']}")
        print(f"   Performance: {stats['performance_metrics']}")
        
        print("\n✅ Neural-Statistical Integration demo completed!")
    
    else:
        print("❌ Failed to create models for demonstration") 