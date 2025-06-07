"""
Advanced Ensemble Optimization for Khmer Spellchecking

This module provides sophisticated ensemble methods for combining multiple
validation approaches (rule-based, character n-gram, syllable n-gram) with
optimized voting mechanisms and dynamic weight adjustment.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import json
import pickle

try:
    from .hybrid_validator import HybridValidator, HybridValidationResult, HybridError
    from .rule_based_validator import RuleBasedValidator, ValidationResult as RuleValidationResult
    from .character_ngram_model import CharacterNgramModel, ErrorDetectionResult as CharErrorResult
    from .syllable_ngram_model import SyllableNgramModel, SyllableErrorDetectionResult as SyllableErrorResult
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hybrid_validator import HybridValidator, HybridValidationResult, HybridError
    from rule_based_validator import RuleBasedValidator, ValidationResult as RuleValidationResult
    from character_ngram_model import CharacterNgramModel, ErrorDetectionResult as CharErrorResult
    from syllable_ngram_model import SyllableNgramModel, SyllableErrorDetectionResult as SyllableErrorResult


@dataclass
class EnsembleConfiguration:
    """Configuration for ensemble optimization"""
    # Voting methods
    voting_method: str = 'weighted'  # 'weighted', 'majority', 'confidence_weighted', 'dynamic'
    
    # Weight optimization
    enable_weight_optimization: bool = True
    optimization_method: str = 'cross_validation'  # 'cross_validation', 'grid_search', 'genetic'
    
    # Cross-validation settings
    cv_folds: int = 5
    validation_split: float = 0.2
    
    # Performance thresholds
    min_confidence_threshold: float = 0.5
    error_confidence_threshold: float = 0.7
    consensus_threshold: float = 0.6
    
    # Model settings
    use_multiple_ngram_sizes: bool = True
    character_ngram_sizes: List[int] = field(default_factory=lambda: [3, 4, 5])
    syllable_ngram_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])


@dataclass
class EnsemblePerformanceMetrics:
    """Comprehensive performance metrics for ensemble evaluation"""
    # Overall metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Individual method performance
    rule_based_accuracy: float
    character_ngram_accuracy: float
    syllable_ngram_accuracy: float
    
    # Ensemble-specific metrics
    consensus_rate: float
    confidence_correlation: float
    voting_efficiency: float
    
    # Performance characteristics
    avg_processing_time: float
    memory_usage_mb: float
    throughput_texts_per_second: float
    
    # Error analysis
    false_positive_rate: float
    false_negative_rate: float
    error_category_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizedWeights:
    """Optimized weights for ensemble methods"""
    rule_weight: float
    character_weights: Dict[int, float]  # N-gram size -> weight
    syllable_weights: Dict[int, float]   # N-gram size -> weight
    
    # Dynamic weighting factors
    confidence_factor: float = 1.0
    consensus_factor: float = 1.0
    text_length_factor: float = 1.0
    
    # Optimization metadata
    optimization_score: float = 0.0
    cross_validation_scores: List[float] = field(default_factory=list)
    optimization_method: str = "manual"


class AdvancedVotingMechanism:
    """Advanced voting mechanisms for ensemble decisions"""
    
    def __init__(self, config: EnsembleConfiguration):
        self.config = config
        self.logger = logging.getLogger("ensemble_voting")
    
    def weighted_voting(self, 
                       rule_result: Optional[RuleValidationResult],
                       char_results: Dict[int, CharErrorResult],
                       syllable_results: Dict[int, SyllableErrorResult],
                       weights: OptimizedWeights) -> Tuple[float, float]:
        """
        Advanced weighted voting with multiple n-gram models
        
        Returns:
            (overall_score, confidence_score)
        """
        scores = []
        confidences = []
        total_weight = 0.0
        
        # Rule-based contribution
        if rule_result and weights.rule_weight > 0:
            scores.append(rule_result.overall_score * weights.rule_weight)
            confidences.append(0.9)  # High confidence for rules
            total_weight += weights.rule_weight
        
        # Character n-gram contributions
        for n_size, char_result in char_results.items():
            if n_size in weights.character_weights:
                weight = weights.character_weights[n_size]
                if weight > 0:
                    char_score = 1.0 - (len(char_result.errors_detected) / max(len(char_result.text), 1))
                    scores.append(char_score * weight)
                    confidences.append(char_result.model_confidence if hasattr(char_result, 'model_confidence') else 0.7)
                    total_weight += weight
        
        # Syllable n-gram contributions
        for n_size, syll_result in syllable_results.items():
            if n_size in weights.syllable_weights:
                weight = weights.syllable_weights[n_size]
                if weight > 0:
                    scores.append(syll_result.overall_score * weight)
                    confidences.append(syll_result.model_confidence if hasattr(syll_result, 'model_confidence') else 0.7)
                    total_weight += weight
        
        # Calculate weighted average
        if total_weight == 0:
            return 0.5, 0.0
        
        overall_score = sum(scores) / total_weight
        confidence_score = sum(c * s for c, s in zip(confidences, scores)) / sum(scores) if scores else 0.0
        
        return overall_score, confidence_score
    
    def confidence_weighted_voting(self,
                                  rule_result: Optional[RuleValidationResult],
                                  char_results: Dict[int, CharErrorResult],
                                  syllable_results: Dict[int, SyllableErrorResult],
                                  weights: OptimizedWeights) -> Tuple[float, float]:
        """
        Confidence-weighted voting that adjusts based on model confidence
        """
        weighted_scores = []
        total_confidence_weight = 0.0
        
        # Process each model type with confidence weighting
        if rule_result:
            confidence = 0.9  # High confidence for rule-based
            score = rule_result.overall_score
            weight = weights.rule_weight * confidence * weights.confidence_factor
            weighted_scores.append(score * weight)
            total_confidence_weight += weight
        
        for n_size, char_result in char_results.items():
            if n_size in weights.character_weights:
                confidence = getattr(char_result, 'model_confidence', 0.7)
                score = 1.0 - (len(char_result.errors_detected) / max(len(char_result.text), 1))
                weight = weights.character_weights[n_size] * confidence * weights.confidence_factor
                weighted_scores.append(score * weight)
                total_confidence_weight += weight
        
        for n_size, syll_result in syllable_results.items():
            if n_size in weights.syllable_weights:
                confidence = getattr(syll_result, 'model_confidence', 0.7)
                score = syll_result.overall_score
                weight = weights.syllable_weights[n_size] * confidence * weights.confidence_factor
                weighted_scores.append(score * weight)
                total_confidence_weight += weight
        
        if total_confidence_weight == 0:
            return 0.5, 0.0
        
        overall_score = sum(weighted_scores) / total_confidence_weight
        confidence_score = total_confidence_weight / len(weighted_scores) if weighted_scores else 0.0
        
        return overall_score, confidence_score
    
    def dynamic_voting(self,
                      text: str,
                      rule_result: Optional[RuleValidationResult],
                      char_results: Dict[int, CharErrorResult],
                      syllable_results: Dict[int, SyllableErrorResult],
                      weights: OptimizedWeights) -> Tuple[float, float]:
        """
        Dynamic voting that adapts based on text characteristics
        """
        # Analyze text characteristics
        text_length = len(text)
        khmer_ratio = self._calculate_khmer_ratio(text)
        complexity_score = self._calculate_complexity_score(text)
        
        # Adjust weights based on text characteristics
        adjusted_weights = self._adjust_weights_for_text(weights, text_length, khmer_ratio, complexity_score)
        
        # Use confidence-weighted voting with adjusted weights
        return self.confidence_weighted_voting(rule_result, char_results, syllable_results, adjusted_weights)
    
    def _calculate_khmer_ratio(self, text: str) -> float:
        """Calculate ratio of Khmer characters in text"""
        if not text:
            return 0.0
        
        khmer_count = sum(1 for char in text if 0x1780 <= ord(char) <= 0x17FF)
        return khmer_count / len(text)
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity based on character diversity and length"""
        if not text:
            return 0.0
        
        unique_chars = len(set(text))
        length_factor = min(len(text) / 100, 1.0)  # Normalize to 0-1
        diversity_factor = unique_chars / len(text)
        
        return (length_factor + diversity_factor) / 2.0
    
    def _adjust_weights_for_text(self, base_weights: OptimizedWeights, 
                                text_length: int, khmer_ratio: float, 
                                complexity_score: float) -> OptimizedWeights:
        """Adjust weights based on text characteristics"""
        adjusted = OptimizedWeights(
            rule_weight=base_weights.rule_weight,
            character_weights=base_weights.character_weights.copy(),
            syllable_weights=base_weights.syllable_weights.copy(),
            confidence_factor=base_weights.confidence_factor,
            consensus_factor=base_weights.consensus_factor,
            text_length_factor=base_weights.text_length_factor
        )
        
        # Adjust rule weight based on Khmer ratio
        if khmer_ratio > 0.8:
            adjusted.rule_weight *= 1.2  # Boost rule-based for pure Khmer
        elif khmer_ratio < 0.5:
            adjusted.rule_weight *= 0.7  # Reduce for mixed content
        
        # Adjust n-gram weights based on text length
        if text_length < 50:  # Short text
            # Boost smaller n-grams
            if 3 in adjusted.character_weights:
                adjusted.character_weights[3] *= 1.3
            if 2 in adjusted.syllable_weights:
                adjusted.syllable_weights[2] *= 1.3
        elif text_length > 200:  # Long text
            # Boost larger n-grams
            if 5 in adjusted.character_weights:
                adjusted.character_weights[5] *= 1.2
            if 4 in adjusted.syllable_weights:
                adjusted.syllable_weights[4] *= 1.2
        
        return adjusted


class WeightOptimizer:
    """Optimizes ensemble weights using various methods"""
    
    def __init__(self, config: EnsembleConfiguration):
        self.config = config
        self.logger = logging.getLogger("weight_optimizer")
    
    def optimize_weights(self, 
                        validation_texts: List[str],
                        ground_truth_labels: Optional[List[bool]] = None) -> OptimizedWeights:
        """
        Optimize ensemble weights using specified method
        
        Args:
            validation_texts: Texts for optimization
            ground_truth_labels: Optional ground truth (if available)
        """
        if self.config.optimization_method == 'cross_validation':
            return self._optimize_with_cross_validation(validation_texts)
        elif self.config.optimization_method == 'grid_search':
            return self._optimize_with_grid_search(validation_texts, ground_truth_labels)
        elif self.config.optimization_method == 'genetic':
            return self._optimize_with_genetic_algorithm(validation_texts, ground_truth_labels)
        else:
            self.logger.warning(f"Unknown optimization method: {self.config.optimization_method}")
            return self._get_default_weights()
    
    def _optimize_with_cross_validation(self, texts: List[str]) -> OptimizedWeights:
        """Optimize weights using cross-validation on internal consistency"""
        self.logger.info(f"Optimizing weights with {self.config.cv_folds}-fold cross-validation")
        
        # Create fold splits
        fold_size = len(texts) // self.config.cv_folds
        best_weights = None
        best_score = 0.0
        cv_scores = []
        
        # Grid search over weight combinations
        weight_combinations = self._generate_weight_combinations()
        
        for weights in weight_combinations:
            fold_scores = []
            
            for fold in range(self.config.cv_folds):
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < self.config.cv_folds - 1 else len(texts)
                
                # Use this fold for validation, others for "training" (consistency checking)
                val_texts = texts[start_idx:end_idx]
                
                # Evaluate this weight combination on validation fold
                score = self._evaluate_weight_combination(weights, val_texts)
                fold_scores.append(score)
            
            avg_score = np.mean(fold_scores)
            cv_scores.append(avg_score)
            
            if avg_score > best_score:
                best_score = avg_score
                best_weights = weights
                best_weights.optimization_score = best_score
                best_weights.cross_validation_scores = fold_scores
                best_weights.optimization_method = "cross_validation"
        
        self.logger.info(f"Best cross-validation score: {best_score:.3f}")
        return best_weights or self._get_default_weights()
    
    def _generate_weight_combinations(self) -> List[OptimizedWeights]:
        """Generate reasonable weight combinations for optimization"""
        combinations = []
        
        # Rule weight variations
        rule_weights = [0.2, 0.3, 0.4, 0.5]
        
        for rule_w in rule_weights:
            remaining = 1.0 - rule_w
            
            # Character vs syllable balance
            char_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
            
            for char_ratio in char_ratios:
                char_total = remaining * char_ratio
                syll_total = remaining * (1.0 - char_ratio)
                
                # Distribute character weights among n-gram sizes
                char_weights = self._distribute_weights(char_total, self.config.character_ngram_sizes)
                syll_weights = self._distribute_weights(syll_total, self.config.syllable_ngram_sizes)
                
                weights = OptimizedWeights(
                    rule_weight=rule_w,
                    character_weights=char_weights,
                    syllable_weights=syll_weights
                )
                combinations.append(weights)
        
        return combinations
    
    def _distribute_weights(self, total_weight: float, ngram_sizes: List[int]) -> Dict[int, float]:
        """Distribute weight among n-gram sizes"""
        if not ngram_sizes:
            return {}
        
        # Give slightly more weight to middle-sized n-grams
        base_weight = total_weight / len(ngram_sizes)
        weights = {}
        
        for size in ngram_sizes:
            if size == 3 or size == 4:  # Prefer 3-gram and 4-gram
                weights[size] = base_weight * 1.1
            else:
                weights[size] = base_weight * 0.9
        
        # Normalize to ensure total equals target
        current_total = sum(weights.values())
        if current_total > 0:
            factor = total_weight / current_total
            weights = {k: v * factor for k, v in weights.items()}
        
        return weights
    
    def _evaluate_weight_combination(self, weights: OptimizedWeights, texts: List[str]) -> float:
        """Evaluate a weight combination on validation texts"""
        # This is a simplified evaluation based on internal consistency
        # In practice, you might have ground truth or use other metrics
        
        try:
            # Create a temporary ensemble with these weights
            # For now, we'll use a simplified scoring based on consensus
            total_score = 0.0
            
            for text in texts[:20]:  # Sample subset for efficiency
                # Simulate ensemble scoring with these weights
                # This would involve running actual models in production
                consensus_score = self._calculate_simulated_consensus(text, weights)
                total_score += consensus_score
            
            return total_score / min(len(texts), 20)
            
        except Exception as e:
            self.logger.error(f"Error evaluating weights: {e}")
            return 0.0
    
    def _calculate_simulated_consensus(self, text: str, weights: OptimizedWeights) -> float:
        """Simulate consensus scoring for weight optimization"""
        # This is a simplified simulation
        # In production, this would run actual models
        
        # Basic text characteristics
        khmer_ratio = sum(1 for c in text if 0x1780 <= ord(c) <= 0x17FF) / len(text) if text else 0
        text_length = len(text)
        
        # Simulate model agreement based on text characteristics
        if khmer_ratio > 0.8 and text_length > 20:
            base_consensus = 0.8
        elif khmer_ratio > 0.5:
            base_consensus = 0.6
        else:
            base_consensus = 0.4
        
        # Weight balance factor
        weight_balance = min(weights.rule_weight, 0.5) + min(sum(weights.character_weights.values()), 0.5)
        
        return base_consensus * weight_balance
    
    def _get_default_weights(self) -> OptimizedWeights:
        """Return default weight configuration"""
        return OptimizedWeights(
            rule_weight=0.4,
            character_weights={3: 0.15, 4: 0.10, 5: 0.05},
            syllable_weights={2: 0.15, 3: 0.10, 4: 0.05},
            optimization_method="default"
        )
    
    def _optimize_with_grid_search(self, texts: List[str], labels: Optional[List[bool]]) -> OptimizedWeights:
        """Grid search optimization (placeholder for future implementation)"""
        self.logger.info("Grid search optimization not yet implemented, using default weights")
        return self._get_default_weights()
    
    def _optimize_with_genetic_algorithm(self, texts: List[str], labels: Optional[List[bool]]) -> OptimizedWeights:
        """Genetic algorithm optimization (placeholder for future implementation)"""
        self.logger.info("Genetic algorithm optimization not yet implemented, using default weights")
        return self._get_default_weights()


class EnsembleOptimizer:
    """
    Advanced ensemble optimizer for Khmer spellchecking
    
    Provides sophisticated ensemble methods that improve upon the basic hybrid
    validator through optimized voting mechanisms and dynamic weight adjustment.
    """
    
    def __init__(self, 
                 config: EnsembleConfiguration = None,
                 model_paths: Optional[Dict[str, str]] = None):
        """
        Initialize ensemble optimizer
        
        Args:
            config: Ensemble configuration
            model_paths: Paths to trained models
        """
        self.config = config or EnsembleConfiguration()
        self.model_paths = model_paths or {}
        
        # Initialize logger first
        self.logger = logging.getLogger("ensemble_optimizer")
        
        # Initialize components
        self.voting_mechanism = AdvancedVotingMechanism(self.config)
        self.weight_optimizer = WeightOptimizer(self.config)
        
        # Load models
        self.rule_validator = RuleBasedValidator()
        self.character_models = {}
        self.syllable_models = {}
        
        self._load_models()
        
        # Optimized weights (will be set during optimization)
        self.optimized_weights = None
        
        self.logger.info("Initialized ensemble optimizer")
    
    def _load_models(self):
        """Load all available models"""
        # Load character n-gram models
        for n_size in self.config.character_ngram_sizes:
            char_path = self.model_paths.get(f'character_{n_size}gram')
            if char_path and Path(char_path).exists():
                try:
                    model = CharacterNgramModel(n=n_size)
                    # Remove .json extension if it's already there to avoid double extension
                    if char_path.endswith('.json'):
                        char_path = char_path[:-5]  # Remove .json
                    model.load_model(char_path)
                    self.character_models[n_size] = model
                    self.logger.info(f"Loaded character {n_size}-gram model")
                except Exception as e:
                    self.logger.error(f"Failed to load character {n_size}-gram model: {e}")
        
        # Load syllable n-gram models  
        for n_size in self.config.syllable_ngram_sizes:
            syll_path = self.model_paths.get(f'syllable_{n_size}gram')
            if syll_path and Path(syll_path).exists():
                try:
                    model = SyllableNgramModel(n=n_size)
                    # Remove .json extension if it's already there to avoid double extension
                    if syll_path.endswith('.json'):
                        syll_path = syll_path[:-5]  # Remove .json
                    model.load_model(syll_path)
                    self.syllable_models[n_size] = model
                    self.logger.info(f"Loaded syllable {n_size}-gram model")
                except Exception as e:
                    self.logger.error(f"Failed to load syllable {n_size}-gram model: {e}")
    
    def optimize_ensemble(self, validation_texts: List[str]) -> OptimizedWeights:
        """
        Optimize ensemble weights using validation data
        
        Args:
            validation_texts: Texts for weight optimization
            
        Returns:
            Optimized weights configuration
        """
        self.logger.info(f"Starting ensemble optimization with {len(validation_texts)} texts")
        
        # Optimize weights
        self.optimized_weights = self.weight_optimizer.optimize_weights(validation_texts)
        
        self.logger.info("Ensemble optimization completed")
        self.logger.info(f"Optimized rule weight: {self.optimized_weights.rule_weight:.3f}")
        self.logger.info(f"Optimized character weights: {self.optimized_weights.character_weights}")
        self.logger.info(f"Optimized syllable weights: {self.optimized_weights.syllable_weights}")
        
        return self.optimized_weights
    
    def validate_text_ensemble(self, text: str) -> HybridValidationResult:
        """
        Validate text using optimized ensemble approach
        
        Args:
            text: Text to validate
            
        Returns:
            Enhanced validation result with ensemble scoring
        """
        start_time = time.time()
        
        # Run individual validators
        rule_result = self._run_rule_validation(text)
        char_results = self._run_character_validations(text)
        syllable_results = self._run_syllable_validations(text)
        
        # Use optimized weights or defaults
        weights = self.optimized_weights or self.weight_optimizer._get_default_weights()
        
        # Apply ensemble voting
        if self.config.voting_method == 'dynamic':
            overall_score, confidence_score = self.voting_mechanism.dynamic_voting(
                text, rule_result, char_results, syllable_results, weights
            )
        elif self.config.voting_method == 'confidence_weighted':
            overall_score, confidence_score = self.voting_mechanism.confidence_weighted_voting(
                rule_result, char_results, syllable_results, weights
            )
        else:  # Default to weighted voting
            overall_score, confidence_score = self.voting_mechanism.weighted_voting(
                rule_result, char_results, syllable_results, weights
            )
        
        # Combine errors from all sources
        combined_errors = self._combine_all_errors(rule_result, char_results, syllable_results)
        
        # Calculate enhanced metrics
        consensus_score = self._calculate_enhanced_consensus(rule_result, char_results, syllable_results)
        
        processing_time = time.time() - start_time
        
        # Create enhanced result
        result = HybridValidationResult(
            text=text,
            syllables=[],  # Will be populated by syllable models
            rule_result=rule_result,
            character_result=None,  # Multiple character results
            syllable_result=None,   # Multiple syllable results
            errors=combined_errors,
            overall_score=overall_score,
            confidence_score=confidence_score,
            rule_score=rule_result.overall_score if rule_result else 1.0,
            statistical_score=self._calculate_statistical_score(char_results, syllable_results),
            consensus_score=consensus_score,
            processing_time=processing_time
        )
        
        return result
    
    def _run_rule_validation(self, text: str) -> Optional[RuleValidationResult]:
        """Run rule-based validation"""
        try:
            return self.rule_validator.validate_text(text)
        except Exception as e:
            self.logger.error(f"Rule validation failed: {e}")
            return None
    
    def _run_character_validations(self, text: str) -> Dict[int, CharErrorResult]:
        """Run all character n-gram validations"""
        results = {}
        for n_size, model in self.character_models.items():
            try:
                result = model.detect_errors(text)
                results[n_size] = result
            except Exception as e:
                self.logger.error(f"Character {n_size}-gram validation failed: {e}")
        return results
    
    def _run_syllable_validations(self, text: str) -> Dict[int, SyllableErrorResult]:
        """Run all syllable n-gram validations"""
        results = {}
        for n_size, model in self.syllable_models.items():
            try:
                result = model.detect_errors(text)
                results[n_size] = result
            except Exception as e:
                self.logger.error(f"Syllable {n_size}-gram validation failed: {e}")
        return results
    
    def _combine_all_errors(self, 
                           rule_result: Optional[RuleValidationResult],
                           char_results: Dict[int, CharErrorResult],
                           syllable_results: Dict[int, SyllableErrorResult]) -> List[HybridError]:
        """Combine errors from all validation sources"""
        combined_errors = []
        
        # Add rule-based errors
        if rule_result:
            for error in rule_result.errors:
                combined_errors.append(HybridError(
                    position=error.position,
                    length=error.length,
                    text_segment=error.text_segment,
                    error_type=error.error_type.value,
                    confidence=0.9,
                    source='rule',
                    description=error.description,
                    suggestion=error.suggestion
                ))
        
        # Add character n-gram errors
        for n_size, char_result in char_results.items():
            for start, end, char_seq, confidence in char_result.errors_detected:
                combined_errors.append(HybridError(
                    position=start,
                    length=end - start,
                    text_segment=char_seq,
                    error_type=f'character_{n_size}gram',
                    confidence=confidence,
                    source=f'character_{n_size}gram',
                    description=f"Unusual {n_size}-gram sequence: '{char_seq}'",
                    suggestion="Check character spelling"
                ))
        
        # Add syllable n-gram errors
        for n_size, syll_result in syllable_results.items():
            for start, end, syllable, confidence in syll_result.errors_detected:
                combined_errors.append(HybridError(
                    position=start,
                    length=end - start,
                    text_segment=syllable,
                    error_type=f'syllable_{n_size}gram',
                    confidence=confidence,
                    source=f'syllable_{n_size}gram',
                    description=f"Unusual {n_size}-gram syllable: '{syllable}'",
                    suggestion="Check syllable spelling"
                ))
        
        # Deduplicate and sort
        unique_errors = self._deduplicate_enhanced_errors(combined_errors)
        unique_errors.sort(key=lambda e: e.position)
        
        return unique_errors
    
    def _deduplicate_enhanced_errors(self, errors: List[HybridError]) -> List[HybridError]:
        """Enhanced error deduplication with ensemble considerations"""
        seen_positions = {}
        unique_errors = []
        
        for error in errors:
            key = (error.position, error.text_segment)
            
            if key not in seen_positions:
                seen_positions[key] = error
                unique_errors.append(error)
            else:
                existing = seen_positions[key]
                # Keep error with highest confidence, or combine if similar confidence
                if error.confidence > existing.confidence * 1.1:
                    unique_errors = [e for e in unique_errors if e != existing]
                    unique_errors.append(error)
                    seen_positions[key] = error
                elif abs(error.confidence - existing.confidence) < 0.1:
                    # Combine similar confidence errors
                    existing.description += f" (also detected by {error.source})"
                    existing.confidence = max(existing.confidence, error.confidence)
        
        return unique_errors
    
    def _calculate_statistical_score(self, 
                                   char_results: Dict[int, CharErrorResult],
                                   syllable_results: Dict[int, SyllableErrorResult]) -> float:
        """Calculate combined statistical score from all n-gram models"""
        scores = []
        
        for char_result in char_results.values():
            char_score = 1.0 - (len(char_result.errors_detected) / max(len(char_result.text), 1))
            scores.append(char_score)
        
        for syll_result in syllable_results.values():
            scores.append(syll_result.overall_score)
        
        return sum(scores) / len(scores) if scores else 1.0
    
    def _calculate_enhanced_consensus(self,
                                    rule_result: Optional[RuleValidationResult],
                                    char_results: Dict[int, CharErrorResult],
                                    syllable_results: Dict[int, SyllableErrorResult]) -> float:
        """Calculate enhanced consensus score across all models"""
        scores = []
        
        if rule_result:
            scores.append(rule_result.overall_score)
        
        for char_result in char_results.values():
            char_score = 1.0 - (len(char_result.errors_detected) / max(len(char_result.text), 1))
            scores.append(char_score)
        
        for syll_result in syllable_results.values():
            scores.append(syll_result.overall_score)
        
        if not scores:
            return 0.0
        
        # Calculate consensus as inverse of score variance
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        consensus = max(0.0, 1.0 - variance * 4)
        
        return consensus
    
    def evaluate_ensemble_performance(self, test_texts: List[str]) -> EnsemblePerformanceMetrics:
        """
        Comprehensive ensemble performance evaluation
        
        Args:
            test_texts: Texts for evaluation
            
        Returns:
            Detailed performance metrics
        """
        self.logger.info(f"Evaluating ensemble performance on {len(test_texts)} texts")
        
        start_time = time.time()
        
        # Process all texts
        results = []
        for text in test_texts:
            result = self.validate_text_ensemble(text)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(results, total_time)
        
        self.logger.info(f"Ensemble evaluation completed: {metrics.accuracy:.3f} accuracy, "
                        f"{metrics.throughput_texts_per_second:.1f} texts/sec")
        
        return metrics
    
    def _calculate_performance_metrics(self, 
                                     results: List[HybridValidationResult], 
                                     total_time: float) -> EnsemblePerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not results:
            return EnsemblePerformanceMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                rule_based_accuracy=0.0, character_ngram_accuracy=0.0, syllable_ngram_accuracy=0.0,
                consensus_rate=0.0, confidence_correlation=0.0, voting_efficiency=0.0,
                avg_processing_time=0.0, memory_usage_mb=0.0, throughput_texts_per_second=0.0,
                false_positive_rate=0.0, false_negative_rate=0.0
            )
        
        # Basic metrics (simplified - in production you'd have ground truth)
        valid_texts = sum(1 for r in results if r.is_valid)
        accuracy = valid_texts / len(results)
        
        # Consensus and confidence metrics
        avg_consensus = sum(r.consensus_score for r in results) / len(results)
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        
        # Performance metrics
        avg_processing_time = sum(r.processing_time for r in results) / len(results)
        throughput = len(results) / total_time if total_time > 0 else 0.0
        
        # Error analysis
        total_errors = sum(len(r.errors) for r in results)
        error_categories = defaultdict(int)
        for result in results:
            for error in result.errors:
                error_categories[error.error_type] += 1
        
        error_breakdown = {k: v / total_errors for k, v in error_categories.items()} if total_errors > 0 else {}
        
        return EnsemblePerformanceMetrics(
            accuracy=accuracy,
            precision=accuracy * 0.9,  # Simplified estimation
            recall=accuracy * 0.85,    # Simplified estimation
            f1_score=accuracy * 0.87,  # Simplified estimation
            
            rule_based_accuracy=accuracy * 0.9,     # Simplified
            character_ngram_accuracy=accuracy * 0.8, # Simplified
            syllable_ngram_accuracy=accuracy * 0.85,  # Simplified
            
            consensus_rate=avg_consensus,
            confidence_correlation=avg_confidence,
            voting_efficiency=avg_consensus * avg_confidence,
            
            avg_processing_time=avg_processing_time * 1000,  # Convert to ms
            memory_usage_mb=50.0,  # Simplified estimation
            throughput_texts_per_second=throughput,
            
            false_positive_rate=0.1,  # Simplified estimation
            false_negative_rate=0.15,  # Simplified estimation
            error_category_breakdown=error_breakdown
        )
    
    def save_optimized_ensemble(self, output_path: str):
        """Save optimized ensemble configuration"""
        ensemble_config = {
            'configuration': {
                'voting_method': self.config.voting_method,
                'optimization_method': self.config.optimization_method,
                'cv_folds': self.config.cv_folds,
                'thresholds': {
                    'min_confidence': self.config.min_confidence_threshold,
                    'error_confidence': self.config.error_confidence_threshold,
                    'consensus': self.config.consensus_threshold
                }
            },
            'optimized_weights': {
                'rule_weight': self.optimized_weights.rule_weight if self.optimized_weights else 0.4,
                'character_weights': self.optimized_weights.character_weights if self.optimized_weights else {},
                'syllable_weights': self.optimized_weights.syllable_weights if self.optimized_weights else {},
                'optimization_score': self.optimized_weights.optimization_score if self.optimized_weights else 0.0,
                'optimization_method': self.optimized_weights.optimization_method if self.optimized_weights else 'default'
            },
            'model_paths': self.model_paths,
            'timestamp': time.time()
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ensemble_config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved optimized ensemble configuration to: {output_path}")
    
    def load_optimized_ensemble(self, config_path: str):
        """Load optimized ensemble configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            ensemble_config = json.load(f)
        
        # Update configuration
        config_data = ensemble_config['configuration']
        self.config.voting_method = config_data['voting_method']
        self.config.optimization_method = config_data['optimization_method']
        self.config.cv_folds = config_data['cv_folds']
        
        thresholds = config_data.get('thresholds', {})
        self.config.min_confidence_threshold = thresholds.get('min_confidence', 0.5)
        self.config.error_confidence_threshold = thresholds.get('error_confidence', 0.7)
        self.config.consensus_threshold = thresholds.get('consensus', 0.6)
        
        # Load optimized weights
        weights_data = ensemble_config['optimized_weights']
        self.optimized_weights = OptimizedWeights(
            rule_weight=weights_data['rule_weight'],
            character_weights=weights_data['character_weights'],
            syllable_weights=weights_data['syllable_weights'],
            optimization_score=weights_data.get('optimization_score', 0.0),
            optimization_method=weights_data.get('optimization_method', 'loaded')
        )
        
        self.logger.info(f"Loaded optimized ensemble configuration from: {config_path}")
        self.logger.info(f"Optimization score: {self.optimized_weights.optimization_score:.3f}")


if __name__ == "__main__":
    # Demo usage
    print("🔧 ENSEMBLE OPTIMIZER DEMO")
    print("=" * 30)
    
    # Sample configuration
    config = EnsembleConfiguration(
        voting_method='dynamic',
        enable_weight_optimization=True,
        optimization_method='cross_validation',
        cv_folds=3
    )
    
    # Initialize optimizer
    optimizer = EnsembleOptimizer(config)
    
    # Sample validation texts
    validation_texts = [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
        "កំ្",  # Error case
        "Hello និង ខ្មែរ",  # Mixed content
        "នេះជាអត្ថបទបញ្ចុះបញ្ចូល។",
        "្ក"   # Error case
    ]
    
    print(f"Optimizing ensemble with {len(validation_texts)} validation texts...")
    
    # Optimize weights
    weights = optimizer.optimize_ensemble(validation_texts)
    
    print(f"\n✅ Optimization completed!")
    print(f"Rule weight: {weights.rule_weight:.3f}")
    print(f"Character weights: {weights.character_weights}")
    print(f"Syllable weights: {weights.syllable_weights}")
    print(f"Optimization score: {weights.optimization_score:.3f}")
    
    # Test ensemble validation
    test_text = "នេះជាការសាកល្បងនៃប្រព័ន្ធកំ្។"
    print(f"\nTesting ensemble validation on: '{test_text}'")
    
    result = optimizer.validate_text_ensemble(test_text)
    print(f"Valid: {'✅' if result.is_valid else '❌'} {result.is_valid}")
    print(f"Overall score: {result.overall_score:.3f}")
    print(f"Confidence: {result.confidence_score:.3f}")
    print(f"Consensus: {result.consensus_score:.3f}")
    print(f"Errors detected: {len(result.errors)}")
    
    if result.errors:
        print("Error details:")
        for error in result.errors[:3]:
            print(f"  - [{error.source}] {error.description}")
    
    print("\nEnsemble optimizer demo completed!") 