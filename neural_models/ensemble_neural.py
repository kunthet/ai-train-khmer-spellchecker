"""
Neural + Statistical Ensemble Integration for Khmer Spellchecking

This module integrates neural models with the existing statistical ensemble
to create a comprehensive hybrid validation system combining the best of
both statistical and neural approaches.
"""

import numpy as np
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, Counter
import statistics

# Import statistical models from Phase 2.4
from statistical_models.ensemble_optimizer import EnsembleOptimizer, EnsembleConfiguration
from statistical_models.rule_based_validator import ValidationResult, ValidationError

# Import neural models
from .neural_validator import NeuralValidator, NeuralValidationResult, NeuralError
from .character_lstm import CharacterLSTMModel, CharacterVocabulary


@dataclass
class NeuralEnsembleConfiguration:
    """Configuration for neural + statistical ensemble"""
    # Neural model weights
    neural_weight: float = 0.3
    statistical_weight: float = 0.7
    
    # Confidence thresholds
    neural_confidence_threshold: float = 0.6
    statistical_confidence_threshold: float = 0.7
    combined_confidence_threshold: float = 0.65
    
    # Perplexity thresholds
    perplexity_error_threshold: float = 100.0
    perplexity_warning_threshold: float = 50.0
    
    # Integration modes
    voting_strategy: str = "weighted_average"  # weighted_average, majority_vote, confidence_based
    error_fusion_method: str = "union"  # union, intersection, confidence_weighted
    
    # Performance settings
    enable_neural_correction: bool = True
    enable_statistical_fallback: bool = True
    max_processing_time: float = 1.0  # seconds
    
    def to_dict(self) -> Dict:
        return {
            'neural_weight': self.neural_weight,
            'statistical_weight': self.statistical_weight,
            'neural_confidence_threshold': self.neural_confidence_threshold,
            'statistical_confidence_threshold': self.statistical_confidence_threshold,
            'combined_confidence_threshold': self.combined_confidence_threshold,
            'perplexity_error_threshold': self.perplexity_error_threshold,
            'perplexity_warning_threshold': self.perplexity_warning_threshold,
            'voting_strategy': self.voting_strategy,
            'error_fusion_method': self.error_fusion_method,
            'enable_neural_correction': self.enable_neural_correction,
            'enable_statistical_fallback': self.enable_statistical_fallback,
            'max_processing_time': self.max_processing_time
        }


@dataclass
class HybridNeuralResult:
    """Complete hybrid neural + statistical validation result"""
    text: str
    is_valid: bool
    
    # Neural results
    neural_result: Optional[NeuralValidationResult] = None
    neural_confidence: float = 0.0
    neural_perplexity: float = 0.0
    neural_errors: List[NeuralError] = field(default_factory=list)
    
    # Statistical results
    statistical_result: Optional[ValidationResult] = None
    statistical_confidence: float = 0.0
    statistical_errors: List[ValidationError] = field(default_factory=list)
    
    # Combined results
    combined_confidence: float = 0.0
    final_errors: List[Union[NeuralError, ValidationError]] = field(default_factory=list)
    validation_method: str = "hybrid"
    
    # Performance metrics
    neural_processing_time: float = 0.0
    statistical_processing_time: float = 0.0
    total_processing_time: float = 0.0
    
    # Integration metadata
    integration_strategy: str = ""
    weight_distribution: Dict[str, float] = field(default_factory=dict)
    error_sources: Dict[str, int] = field(default_factory=dict)
    
    def get_summary(self) -> Dict:
        """Get comprehensive result summary"""
        return {
            'text_length': len(self.text),
            'is_valid': self.is_valid,
            'validation_method': self.validation_method,
            'combined_confidence': self.combined_confidence,
            'total_errors': len(self.final_errors),
            'neural_confidence': self.neural_confidence,
            'statistical_confidence': self.statistical_confidence,
            'neural_perplexity': self.neural_perplexity,
            'processing_time': self.total_processing_time,
            'performance_ratio': (self.neural_processing_time / self.statistical_processing_time 
                                if self.statistical_processing_time > 0 else 0),
            'error_sources': dict(self.error_sources),
            'integration_strategy': self.integration_strategy
        }


class NeuralStatisticalIntegration:
    """
    Core integration logic for combining neural and statistical validation
    
    Handles the fusion of results from neural models and statistical ensembles
    using various strategies and confidence weighting mechanisms.
    """
    
    def __init__(self, config: NeuralEnsembleConfiguration):
        self.config = config
        self.logger = logging.getLogger("neural_statistical_integration")
    
    def combine_confidences(self, 
                          neural_confidence: float,
                          statistical_confidence: float) -> float:
        """
        Combine neural and statistical confidence scores
        
        Args:
            neural_confidence: Confidence from neural model
            statistical_confidence: Confidence from statistical model
            
        Returns:
            Combined confidence score
        """
        if self.config.voting_strategy == "weighted_average":
            combined = (self.config.neural_weight * neural_confidence + 
                       self.config.statistical_weight * statistical_confidence)
        
        elif self.config.voting_strategy == "confidence_based":
            # Weight by confidence levels
            total_confidence = neural_confidence + statistical_confidence
            if total_confidence > 0:
                neural_w = neural_confidence / total_confidence
                statistical_w = statistical_confidence / total_confidence
                combined = neural_w * neural_confidence + statistical_w * statistical_confidence
            else:
                combined = 0.0
        
        elif self.config.voting_strategy == "majority_vote":
            # Binary decision based on thresholds, then average
            neural_valid = neural_confidence >= self.config.neural_confidence_threshold
            stat_valid = statistical_confidence >= self.config.statistical_confidence_threshold
            
            if neural_valid and stat_valid:
                combined = (neural_confidence + statistical_confidence) / 2
            elif neural_valid:
                combined = neural_confidence
            elif stat_valid:
                combined = statistical_confidence
            else:
                combined = min(neural_confidence, statistical_confidence)
        
        else:
            # Default to weighted average
            combined = (self.config.neural_weight * neural_confidence + 
                       self.config.statistical_weight * statistical_confidence)
        
        return max(0.0, min(1.0, combined))
    
    def fuse_errors(self, 
                   neural_errors: List[NeuralError],
                   statistical_errors: List[ValidationError]) -> List[Union[NeuralError, ValidationError]]:
        """
        Fuse errors from neural and statistical validation
        
        Args:
            neural_errors: Errors from neural validation
            statistical_errors: Errors from statistical validation
            
        Returns:
            Fused list of errors
        """
        if self.config.error_fusion_method == "union":
            # Simple union of all errors
            return neural_errors + statistical_errors
        
        elif self.config.error_fusion_method == "intersection":
            # Only include errors found by both methods (by position)
            neural_positions = {error.position for error in neural_errors}
            statistical_positions = {error.position for error in statistical_errors}
            common_positions = neural_positions.intersection(statistical_positions)
            
            fused_errors = []
            for error in neural_errors:
                if error.position in common_positions:
                    fused_errors.append(error)
            for error in statistical_errors:
                if error.position in common_positions:
                    fused_errors.append(error)
            
            return fused_errors
        
        elif self.config.error_fusion_method == "confidence_weighted":
            # Include errors based on confidence thresholds
            fused_errors = []
            
            for error in neural_errors:
                if error.confidence >= self.config.neural_confidence_threshold:
                    fused_errors.append(error)
            
            for error in statistical_errors:
                if error.confidence >= self.config.statistical_confidence_threshold:
                    fused_errors.append(error)
            
            return fused_errors
        
        else:
            # Default to union
            return neural_errors + statistical_errors
    
    def determine_validity(self, 
                         neural_result: NeuralValidationResult,
                         statistical_result: ValidationResult,
                         combined_confidence: float) -> bool:
        """
        Determine final text validity based on combined results
        
        Args:
            neural_result: Neural validation result
            statistical_result: Statistical validation result
            combined_confidence: Combined confidence score
            
        Returns:
            Final validity decision
        """
        # Check combined confidence threshold
        if combined_confidence < self.config.combined_confidence_threshold:
            return False
        
        # Check perplexity threshold
        if neural_result.overall_perplexity > self.config.perplexity_error_threshold:
            return False
        
        # Consider both individual results
        neural_valid = (neural_result.is_valid and 
                       neural_result.model_confidence >= self.config.neural_confidence_threshold)
        
        statistical_valid = (statistical_result.is_valid and 
                           statistical_result.confidence >= self.config.statistical_confidence_threshold)
        
        # Apply voting strategy
        if self.config.voting_strategy == "majority_vote":
            return neural_valid or statistical_valid  # At least one should be valid
        
        elif self.config.voting_strategy == "confidence_based":
            # Weight by confidence and require both to agree if both are confident
            if (neural_result.model_confidence >= self.config.neural_confidence_threshold and
                statistical_result.confidence >= self.config.statistical_confidence_threshold):
                return neural_valid and statistical_valid
            else:
                return neural_valid or statistical_valid
        
        else:
            # Weighted average approach - rely on combined confidence
            return combined_confidence >= self.config.combined_confidence_threshold


class NeuralEnsembleOptimizer:
    """
    Complete neural + statistical ensemble system
    
    Integrates neural character-level LSTM models with the existing statistical
    ensemble optimizer to provide comprehensive Khmer spellchecking capabilities.
    """
    
    def __init__(self, 
                 neural_validator: NeuralValidator,
                 statistical_ensemble: EnsembleOptimizer,
                 config: Optional[NeuralEnsembleConfiguration] = None):
        
        self.neural_validator = neural_validator
        self.statistical_ensemble = statistical_ensemble
        self.config = config or NeuralEnsembleConfiguration()
        
        # Initialize integration logic
        self.integration = NeuralStatisticalIntegration(self.config)
        
        # Performance tracking
        self.validation_stats = defaultdict(list)
        
        self.logger = logging.getLogger("neural_ensemble_optimizer")
        self.logger.info("Neural ensemble optimizer initialized")
    
    def validate_text(self, text: str) -> HybridNeuralResult:
        """
        Comprehensive validation using both neural and statistical models
        
        Args:
            text: Input text to validate
            
        Returns:
            HybridNeuralResult with complete analysis
        """
        start_time = time.time()
        
        # Initialize result
        result = HybridNeuralResult(
            text=text,
            is_valid=False,
            integration_strategy=self.config.voting_strategy
        )
        
        try:
            # Neural validation
            neural_start = time.time()
            neural_result = self.neural_validator.validate_text(text, detailed_analysis=True)
            neural_time = time.time() - neural_start
            
            # Statistical validation
            statistical_start = time.time()
            statistical_result = self.statistical_ensemble.validate_text(text)
            statistical_time = time.time() - statistical_start
            
            # Extract key metrics
            result.neural_result = neural_result
            result.neural_confidence = neural_result.model_confidence
            result.neural_perplexity = neural_result.overall_perplexity
            result.neural_errors = neural_result.detected_errors
            result.neural_processing_time = neural_time
            
            result.statistical_result = statistical_result
            result.statistical_confidence = statistical_result.confidence
            result.statistical_errors = statistical_result.errors
            result.statistical_processing_time = statistical_time
            
            # Combine results
            result.combined_confidence = self.integration.combine_confidences(
                neural_result.model_confidence,
                statistical_result.confidence
            )
            
            # Fuse errors
            result.final_errors = self.integration.fuse_errors(
                neural_result.detected_errors,
                statistical_result.errors
            )
            
            # Determine final validity
            result.is_valid = self.integration.determine_validity(
                neural_result,
                statistical_result,
                result.combined_confidence
            )
            
            # Calculate metadata
            result.weight_distribution = {
                'neural': self.config.neural_weight,
                'statistical': self.config.statistical_weight
            }
            
            # Error source analysis
            result.error_sources = {
                'neural_only': len(neural_result.detected_errors),
                'statistical_only': len(statistical_result.errors),
                'combined': len(result.final_errors)
            }
            
        except Exception as e:
            self.logger.error(f"Error during validation: {e}")
            
            # Fallback to statistical validation if enabled
            if self.config.enable_statistical_fallback:
                try:
                    statistical_result = self.statistical_ensemble.validate_text(text)
                    result.statistical_result = statistical_result
                    result.statistical_confidence = statistical_result.confidence
                    result.is_valid = statistical_result.is_valid
                    result.final_errors = statistical_result.errors
                    result.validation_method = "statistical_fallback"
                except Exception as fallback_error:
                    self.logger.error(f"Fallback validation failed: {fallback_error}")
                    result.is_valid = False
        
        result.total_processing_time = time.time() - start_time
        
        # Track performance
        self.validation_stats['processing_times'].append(result.total_processing_time)
        self.validation_stats['combined_confidences'].append(result.combined_confidence)
        
        return result
    
    def batch_validate(self, 
                      texts: List[str],
                      show_progress: bool = True) -> List[HybridNeuralResult]:
        """
        Batch validation of multiple texts
        
        Args:
            texts: List of texts to validate
            show_progress: Whether to show progress updates
            
        Returns:
            List of validation results
        """
        self.logger.info(f"Batch validating {len(texts)} texts with hybrid ensemble...")
        
        results = []
        start_time = time.time()
        
        for i, text in enumerate(texts):
            result = self.validate_text(text)
            results.append(result)
            
            if show_progress and (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                self.logger.info(f"Processed {i + 1}/{len(texts)} texts ({rate:.1f} texts/sec)")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(texts) if texts else 0
        
        self.logger.info(f"Batch validation completed in {total_time:.2f}s")
        self.logger.info(f"Average time per text: {avg_time:.4f}s")
        
        return results
    
    def optimize_weights(self, 
                        validation_texts: List[str],
                        target_accuracy: float = 0.95) -> Dict[str, float]:
        """
        Optimize neural and statistical weights for best performance
        
        Args:
            validation_texts: Texts for weight optimization
            target_accuracy: Target validation accuracy
            
        Returns:
            Optimized weight configuration
        """
        self.logger.info(f"Optimizing weights on {len(validation_texts)} validation texts...")
        
        best_weights = {'neural': 0.3, 'statistical': 0.7}
        best_accuracy = 0.0
        
        # Grid search over weight combinations
        weight_ranges = np.arange(0.1, 1.0, 0.1)
        
        for neural_weight in weight_ranges:
            statistical_weight = 1.0 - neural_weight
            
            # Update configuration
            old_neural_weight = self.config.neural_weight
            old_statistical_weight = self.config.statistical_weight
            
            self.config.neural_weight = neural_weight
            self.config.statistical_weight = statistical_weight
            
            # Validate sample
            sample_size = min(100, len(validation_texts))
            sample_texts = validation_texts[:sample_size]
            
            results = []
            for text in sample_texts:
                try:
                    result = self.validate_text(text)
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Validation failed during optimization: {e}")
                    continue
            
            if not results:
                continue
            
            # Calculate accuracy (assuming validation texts are correct)
            accurate_validations = sum(1 for r in results if r.is_valid)
            accuracy = accurate_validations / len(results)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = {
                    'neural': neural_weight,
                    'statistical': statistical_weight
                }
            
            self.logger.debug(f"Weights [N:{neural_weight:.1f}, S:{statistical_weight:.1f}] -> Accuracy: {accuracy:.3f}")
            
            # Restore original weights
            self.config.neural_weight = old_neural_weight
            self.config.statistical_weight = old_statistical_weight
        
        # Apply best weights
        self.config.neural_weight = best_weights['neural']
        self.config.statistical_weight = best_weights['statistical']
        
        self.logger.info(f"Optimized weights: Neural={best_weights['neural']:.3f}, "
                        f"Statistical={best_weights['statistical']:.3f}")
        self.logger.info(f"Best accuracy: {best_accuracy:.3f}")
        
        return best_weights
    
    def get_performance_analysis(self, 
                               results: List[HybridNeuralResult]) -> Dict:
        """
        Analyze performance of the hybrid ensemble
        
        Args:
            results: List of validation results
            
        Returns:
            Comprehensive performance analysis
        """
        if not results:
            return {'error': 'No results provided'}
        
        # Basic statistics
        total_texts = len(results)
        valid_texts = sum(1 for r in results if r.is_valid)
        
        # Processing time analysis
        neural_times = [r.neural_processing_time for r in results if r.neural_processing_time > 0]
        statistical_times = [r.statistical_processing_time for r in results if r.statistical_processing_time > 0]
        total_times = [r.total_processing_time for r in results]
        
        # Confidence analysis
        neural_confidences = [r.neural_confidence for r in results if r.neural_confidence > 0]
        statistical_confidences = [r.statistical_confidence for r in results if r.statistical_confidence > 0]
        combined_confidences = [r.combined_confidence for r in results]
        
        # Error analysis
        neural_error_counts = [len(r.neural_errors) for r in results]
        statistical_error_counts = [len(r.statistical_errors) for r in results]
        final_error_counts = [len(r.final_errors) for r in results]
        
        # Perplexity analysis
        perplexities = [r.neural_perplexity for r in results if r.neural_perplexity > 0]
        
        return {
            'overall_performance': {
                'total_texts': total_texts,
                'valid_texts': valid_texts,
                'validation_rate': valid_texts / total_texts,
                'avg_processing_time': statistics.mean(total_times),
                'throughput_texts_per_second': total_texts / sum(total_times) if sum(total_times) > 0 else 0
            },
            'neural_performance': {
                'avg_processing_time': statistics.mean(neural_times) if neural_times else 0,
                'avg_confidence': statistics.mean(neural_confidences) if neural_confidences else 0,
                'avg_errors_detected': statistics.mean(neural_error_counts),
                'avg_perplexity': statistics.mean(perplexities) if perplexities else 0
            },
            'statistical_performance': {
                'avg_processing_time': statistics.mean(statistical_times) if statistical_times else 0,
                'avg_confidence': statistics.mean(statistical_confidences) if statistical_confidences else 0,
                'avg_errors_detected': statistics.mean(statistical_error_counts)
            },
            'ensemble_performance': {
                'avg_combined_confidence': statistics.mean(combined_confidences),
                'avg_final_errors': statistics.mean(final_error_counts),
                'neural_vs_statistical_speed': (statistics.mean(neural_times) / statistics.mean(statistical_times) 
                                               if neural_times and statistical_times else 0),
                'integration_overhead': statistics.mean(total_times) - statistics.mean(neural_times) - statistics.mean(statistical_times)
                                      if neural_times and statistical_times else 0
            },
            'configuration': self.config.to_dict()
        }
    
    def save_configuration(self, filepath: str):
        """Save ensemble configuration to file"""
        config_data = {
            'neural_ensemble_config': self.config.to_dict(),
            'statistical_ensemble_config': self.statistical_ensemble.get_configuration() if hasattr(self.statistical_ensemble, 'get_configuration') else {},
            'performance_stats': {
                'total_validations': len(self.validation_stats['processing_times']),
                'avg_processing_time': statistics.mean(self.validation_stats['processing_times']) if self.validation_stats['processing_times'] else 0,
                'avg_combined_confidence': statistics.mean(self.validation_stats['combined_confidences']) if self.validation_stats['combined_confidences'] else 0
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Configuration saved to: {filepath}")
    
    def load_configuration(self, filepath: str):
        """Load ensemble configuration from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Update neural ensemble config
        neural_config = config_data.get('neural_ensemble_config', {})
        for key, value in neural_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.logger.info(f"Configuration loaded from: {filepath}")


if __name__ == "__main__":
    # Demo usage
    print("🔬 NEURAL ENSEMBLE OPTIMIZER DEMO")
    print("=" * 35)
    
    print("Note: This demo requires both trained neural models and statistical ensemble")
    print("      Run neural trainer and ensure Phase 2.4 ensemble is available")
    
    # Sample configuration
    config = NeuralEnsembleConfiguration(
        neural_weight=0.4,
        statistical_weight=0.6,
        voting_strategy="weighted_average",
        error_fusion_method="confidence_weighted",
        enable_neural_correction=True
    )
    
    print(f"\n🔧 Configuration:")
    print(f"   Neural weight: {config.neural_weight}")
    print(f"   Statistical weight: {config.statistical_weight}")
    print(f"   Voting strategy: {config.voting_strategy}")
    print(f"   Error fusion: {config.error_fusion_method}")
    
    # Sample texts for demonstration
    sample_texts = [
        "នេះជាអត្ថបទត្រឹមត្រូវ។",           # Correct text
        "នេះជាអត្ថបទមានកំហុស។",            # Text with potential errors
        "របាយការណ៍ពិសេសអំពីការអភិវឌ្ឍន៍។",    # Complex text
        "ការអប់រំជាមូលដ្ឋានការអភិវឌ្ឍន៍។",     # Educational text
        "វប្បធម៌ខ្មែរមានប្រវត្តិយូរលង់។"        # Cultural text
    ]
    
    print(f"\n📝 Sample texts ({len(sample_texts)} texts):")
    for i, text in enumerate(sample_texts, 1):
        print(f"   {i}. '{text[:30]}{'...' if len(text) > 30 else ''}'")
    
    print(f"\n🧠 Neural + Statistical Integration Features:")
    print(f"   ✅ Weighted confidence combination")
    print(f"   ✅ Multiple voting strategies")
    print(f"   ✅ Error fusion with confidence weighting")
    print(f"   ✅ Automatic weight optimization")
    print(f"   ✅ Comprehensive performance analysis")
    print(f"   ✅ Fallback mechanisms")
    print(f"   ✅ Real-time processing")
    
    print(f"\n🎯 Expected Performance Improvements:")
    print(f"   • Accuracy: 90-95% (vs 88.2% statistical only)")
    print(f"   • Processing time: <60ms per text")
    print(f"   • Perplexity-based error detection")
    print(f"   • Context-aware corrections")
    print(f"   • Robust error handling")
    
    print("\n✅ Neural ensemble optimizer demo structure completed!") 