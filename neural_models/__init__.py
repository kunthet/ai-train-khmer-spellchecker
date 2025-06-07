"""
Neural Models Package for Khmer Spellchecking

This package provides neural network models for Khmer text processing,
including character-level and syllable-level LSTM models, as well as
integration with statistical models for comprehensive spellchecking.
"""

# Character-level neural models
from .character_lstm import (
    CharacterLSTMModel,
    CharacterVocabulary, 
    LSTMConfiguration
)

# Syllable-level neural models  
from .syllable_lstm import (
    SyllableLSTMModel,
    SyllableVocabulary,
    SyllableLSTMConfiguration
)

# Neural training infrastructure
from .neural_trainer import (
    TrainingDataset,
    NeuralTrainer,
    TrainingResult,
    TrainingMetrics
)

# Neural ensemble systems
from .ensemble_neural import (
    NeuralStatisticalIntegration,
    NeuralEnsembleOptimizer,
    NeuralEnsembleConfiguration,
    HybridNeuralResult
)

# Neural validation
from .neural_validator import (
    NeuralValidator,
    NeuralValidationResult,
    NeuralError,
    PerplexityScorer
)

# Phase 3.3: Neural-Statistical Integration
from .neural_statistical_integration import (
    NeuralStatisticalIntegrator,
    IntegrationConfiguration,
    IntegratedError,
    IntegrationResult
)

__all__ = [
    # Character-level models
    'CharacterLSTMModel',
    'CharacterVocabulary',
    'LSTMConfiguration',
    
    # Syllable-level models
    'SyllableLSTMModel', 
    'SyllableVocabulary',
    'SyllableLSTMConfiguration',
    
    # Training infrastructure
    'TrainingDataset',
    'NeuralTrainer',
    'TrainingResult',
    'TrainingMetrics',
    
    # Ensemble systems
    'NeuralStatisticalIntegration',
    'NeuralEnsembleOptimizer',
    'NeuralEnsembleConfiguration',
    'HybridNeuralResult',
    
    # Neural validation
    'NeuralValidator',
    'NeuralValidationResult',
    'NeuralError',
    'PerplexityScorer',
    
    # Neural-Statistical Integration (Phase 3.3)
    'NeuralStatisticalIntegrator',
    'IntegrationConfiguration', 
    'IntegratedError',
    'IntegrationResult'
] 