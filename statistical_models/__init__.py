"""
Statistical Models Package for Khmer Spellchecking

This package provides statistical language models for Khmer text analysis
and spellchecking, including character-level and syllable-level n-gram models,
rule-based validation, and hybrid approaches.
"""

# Smoothing techniques
from .smoothing_techniques import (
    BaseSmoothingTechnique,
    LaplaceSmoothing,
    GoodTuringSmoothing,
    SimpleBackoffSmoothing,
    SmoothingMethod,
    create_smoothing_technique,
    compare_smoothing_methods
)

# Character-level models
from .character_ngram_model import (
    CharacterNgramModel,
    NgramModelTrainer,
    ErrorDetectionResult,
    CharacterModelStatistics
)

# Syllable-level models
from .syllable_ngram_model import (
    SyllableNgramModel,
    SyllableNgramModelTrainer,
    SyllableErrorDetectionResult,
    SyllableModelStatistics
)

# Rule-based validation
from .rule_based_validator import (
    RuleBasedValidator,
    KhmerSyllableValidator,
    KhmerSequenceValidator,
    KhmerUnicodeRanges,
    ValidationError,
    ValidationResult,
    ValidationErrorType
)

# Hybrid validation
from .hybrid_validator import (
    HybridValidator,
    HybridError,
    HybridValidationResult
)

# Import ensemble optimization components
from .ensemble_optimizer import (
    EnsembleOptimizer,
    EnsembleConfiguration,
    EnsemblePerformanceMetrics,
    OptimizedWeights,
    AdvancedVotingMechanism,
    WeightOptimizer
)

__all__ = [
    # Smoothing techniques
    'BaseSmoothingTechnique',
    'LaplaceSmoothing',
    'GoodTuringSmoothing',
    'SimpleBackoffSmoothing',
    'SmoothingMethod',
    'create_smoothing_technique',
    'compare_smoothing_methods',
    
    # Character-level models
    'CharacterNgramModel',
    'NgramModelTrainer',
    'ErrorDetectionResult',
    'CharacterModelStatistics',
    
    # Syllable-level models
    'SyllableNgramModel',
    'SyllableNgramModelTrainer',
    'SyllableErrorDetectionResult',
    'SyllableModelStatistics',
    
    # Rule-based validation
    'RuleBasedValidator',
    'KhmerSyllableValidator',
    'KhmerSequenceValidator',
    'KhmerUnicodeRanges',
    'ValidationError',
    'ValidationResult',
    'ValidationErrorType',
    
    # Hybrid validation
    'HybridValidator',
    'HybridError',
    'HybridValidationResult',
    
    # Ensemble optimization
    'EnsembleOptimizer',
    'EnsembleConfiguration',
    'EnsemblePerformanceMetrics',
    'OptimizedWeights',
    'AdvancedVotingMechanism',
    'WeightOptimizer',
] 