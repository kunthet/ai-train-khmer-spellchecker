"""
Neural Validation and Error Detection for Khmer Text

This module provides neural-based validation using perplexity scoring,
error detection, and integration with statistical models for comprehensive
Khmer spellchecking.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics

from .character_lstm import CharacterLSTMModel, CharacterVocabulary


@dataclass
class NeuralError:
    """Neural error detection result"""
    position: int
    character: str
    error_type: str
    perplexity_score: float
    confidence: float
    suggested_corrections: List[str] = field(default_factory=list)
    context_window: str = ""
    neural_source: str = "character_lstm"
    
    def to_dict(self) -> Dict:
        return {
            'position': self.position,
            'character': self.character,
            'error_type': self.error_type,
            'perplexity_score': self.perplexity_score,
            'confidence': self.confidence,
            'suggested_corrections': self.suggested_corrections,
            'context_window': self.context_window,
            'neural_source': self.neural_source
        }


@dataclass
class NeuralValidationResult:
    """Complete neural validation result"""
    text: str
    is_valid: bool
    overall_perplexity: float
    character_perplexities: List[float]
    detected_errors: List[NeuralError]
    processing_time: float
    model_confidence: float
    text_complexity: Dict = field(default_factory=dict)
    
    def get_error_summary(self) -> Dict:
        """Get summary of detected errors"""
        if not self.detected_errors:
            return {'total_errors': 0}
        
        error_types = Counter(error.error_type for error in self.detected_errors)
        avg_perplexity = statistics.mean(error.perplexity_score for error in self.detected_errors)
        avg_confidence = statistics.mean(error.confidence for error in self.detected_errors)
        
        return {
            'total_errors': len(self.detected_errors),
            'error_types': dict(error_types),
            'avg_perplexity': avg_perplexity,
            'avg_confidence': avg_confidence,
            'error_positions': [error.position for error in self.detected_errors]
        }


class PerplexityScorer:
    """
    Perplexity-based scoring for character sequences
    
    Calculates character-level and sequence-level perplexity scores
    for error detection and text quality assessment.
    """
    
    def __init__(self, 
                 model: CharacterLSTMModel,
                 vocabulary: CharacterVocabulary,
                 device: str = 'cpu'):
        self.model = model
        self.vocabulary = vocabulary
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        self.logger = logging.getLogger("perplexity_scorer")
    
    def calculate_character_perplexities(self, 
                                       text: str,
                                       context_window: int = 20) -> List[float]:
        """
        Calculate perplexity for each character in the text
        
        Args:
            text: Input text
            context_window: Size of context window for prediction
            
        Returns:
            List of perplexity scores for each character
        """
        if len(text) < 2:
            return [1.0] * len(text)  # Default perplexity for very short text
        
        # Encode text
        encoded_text = self.vocabulary.encode_text(text, max_length=len(text))
        perplexities = []
        
        with torch.no_grad():
            for i in range(1, len(encoded_text)):
                # Get context window
                start_idx = max(0, i - context_window)
                context = encoded_text[start_idx:i]
                target_char = encoded_text[i]
                
                # Pad context to fixed length
                if len(context) < context_window:
                    pad_idx = self.vocabulary.char_to_idx.get('<PAD>', 0)
                    context = [pad_idx] * (context_window - len(context)) + context
                else:
                    context = context[-context_window:]
                
                # Convert to tensor
                context_tensor = torch.tensor([context], dtype=torch.long, device=self.device)
                
                # Get model prediction
                logits = self.model(context_tensor)
                probs = F.softmax(logits, dim=-1)
                
                # Calculate perplexity for target character
                target_prob = probs[0, target_char].item()
                
                # Avoid log(0) by setting minimum probability
                target_prob = max(target_prob, 1e-10)
                character_perplexity = 1.0 / target_prob
                
                perplexities.append(character_perplexity)
        
        # Handle first character (no context)
        if perplexities:
            perplexities.insert(0, perplexities[0])  # Use same as second character
        else:
            perplexities = [1.0] * len(text)
        
        return perplexities
    
    def calculate_sequence_perplexity(self, text: str) -> float:
        """
        Calculate overall sequence perplexity
        
        Args:
            text: Input text
            
        Returns:
            Sequence perplexity score
        """
        char_perplexities = self.calculate_character_perplexities(text)
        
        if not char_perplexities:
            return 1.0
        
        # Geometric mean of character perplexities
        log_perplexities = [np.log(p) for p in char_perplexities if p > 0]
        
        if not log_perplexities:
            return 1.0
        
        avg_log_perplexity = np.mean(log_perplexities)
        sequence_perplexity = np.exp(avg_log_perplexity)
        
        return float(sequence_perplexity)
    
    def identify_anomalous_characters(self, 
                                    text: str,
                                    threshold_multiplier: float = 2.0) -> List[Tuple[int, float]]:
        """
        Identify characters with anomalously high perplexity
        
        Args:
            text: Input text
            threshold_multiplier: Multiplier for anomaly threshold
            
        Returns:
            List of (position, perplexity) for anomalous characters
        """
        char_perplexities = self.calculate_character_perplexities(text)
        
        if len(char_perplexities) < 3:
            return []
        
        # Calculate threshold based on statistics
        median_perplexity = statistics.median(char_perplexities)
        q75 = np.percentile(char_perplexities, 75)
        threshold = median_perplexity + threshold_multiplier * (q75 - median_perplexity)
        
        # Find anomalous characters
        anomalous = []
        for i, perplexity in enumerate(char_perplexities):
            if perplexity > threshold:
                anomalous.append((i, perplexity))
        
        return anomalous


class NeuralValidator:
    """
    Neural-based text validator using character-level LSTM models
    
    Provides comprehensive validation including:
    - Perplexity-based error detection
    - Character-level anomaly detection
    - Context-aware correction suggestions
    - Integration with statistical models
    """
    
    def __init__(self, 
                 model: CharacterLSTMModel,
                 vocabulary: CharacterVocabulary,
                 device: str = 'cpu'):
        self.model = model
        self.vocabulary = vocabulary
        self.device = device
        
        # Initialize components
        self.perplexity_scorer = PerplexityScorer(model, vocabulary, device)
        
        # Configuration
        self.anomaly_threshold_multiplier = 2.0
        self.context_window_size = 15
        self.min_confidence_threshold = 0.1
        self.max_corrections_per_error = 3
        
        self.logger = logging.getLogger("neural_validator")
        self.logger.info("Neural validator initialized")
    
    def _calculate_text_complexity(self, text: str) -> Dict:
        """Calculate text complexity metrics"""
        if not text:
            return {'length': 0, 'khmer_ratio': 0.0, 'unique_chars': 0}
        
        khmer_chars = sum(1 for char in text 
                         if 0x1780 <= ord(char) <= 0x17FF)
        unique_chars = len(set(text))
        
        return {
            'length': len(text),
            'khmer_ratio': khmer_chars / len(text),
            'unique_chars': unique_chars,
            'char_diversity': unique_chars / len(text) if len(text) > 0 else 0
        }
    
    def _generate_corrections(self, 
                            text: str, 
                            error_position: int,
                            n_suggestions: int = 3) -> List[str]:
        """
        Generate correction suggestions for detected errors
        
        Args:
            text: Input text
            error_position: Position of error
            n_suggestions: Number of suggestions to generate
            
        Returns:
            List of suggested corrections
        """
        if error_position >= len(text) or error_position < 0:
            return []
        
        suggestions = []
        
        # Get context around error
        context_start = max(0, error_position - self.context_window_size)
        context_end = min(len(text), error_position + self.context_window_size)
        
        # Encode context before error position
        context_text = text[context_start:error_position]
        encoded_context = self.vocabulary.encode_text(context_text, max_length=len(context_text))
        
        if len(encoded_context) < 5:
            return []  # Need minimum context
        
        # Use last few characters as context for prediction
        context_window = encoded_context[-self.context_window_size:]
        
        # Pad context if needed
        if len(context_window) < self.context_window_size:
            pad_idx = self.vocabulary.char_to_idx.get('<PAD>', 0)
            padding = [pad_idx] * (self.context_window_size - len(context_window))
            context_window = padding + context_window
        
        # Get model predictions
        with torch.no_grad():
            context_tensor = torch.tensor([context_window], dtype=torch.long, device=self.device)
            logits = self.model(context_tensor)
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs[0], k=min(20, self.vocabulary.vocabulary_size))
            
            # Convert to characters and filter
            for prob, idx in zip(top_probs, top_indices):
                char_idx = idx.item()
                confidence = prob.item()
                
                if confidence < self.min_confidence_threshold:
                    continue
                
                char = self.vocabulary.idx_to_char.get(char_idx, '')
                
                # Filter special tokens and current character
                if (char and 
                    char not in self.vocabulary.config.special_tokens and
                    char != text[error_position] and
                    len(suggestions) < n_suggestions):
                    suggestions.append(char)
        
        return suggestions
    
    def validate_text(self, 
                     text: str,
                     detailed_analysis: bool = True) -> NeuralValidationResult:
        """
        Comprehensive neural validation of text
        
        Args:
            text: Input text to validate
            detailed_analysis: Whether to perform detailed character-level analysis
            
        Returns:
            NeuralValidationResult with complete analysis
        """
        start_time = time.time()
        
        if not text or not text.strip():
            return NeuralValidationResult(
                text=text,
                is_valid=True,
                overall_perplexity=1.0,
                character_perplexities=[],
                detected_errors=[],
                processing_time=0.0,
                model_confidence=1.0,
                text_complexity={}
            )
        
        # Calculate text complexity
        complexity = self._calculate_text_complexity(text)
        
        # Calculate perplexities
        overall_perplexity = self.perplexity_scorer.calculate_sequence_perplexity(text)
        
        character_perplexities = []
        detected_errors = []
        
        if detailed_analysis:
            # Character-level analysis
            character_perplexities = self.perplexity_scorer.calculate_character_perplexities(text)
            
            # Identify anomalous characters
            anomalous_chars = self.perplexity_scorer.identify_anomalous_characters(
                text, self.anomaly_threshold_multiplier
            )
            
            # Create error objects for anomalous characters
            for position, perplexity in anomalous_chars:
                if position < len(text):
                    character = text[position]
                    
                    # Generate corrections
                    corrections = self._generate_corrections(
                        text, position, self.max_corrections_per_error
                    )
                    
                    # Calculate confidence (inverse of normalized perplexity)
                    max_perplexity = max(character_perplexities) if character_perplexities else perplexity
                    confidence = 1.0 - (perplexity / max_perplexity) if max_perplexity > 0 else 0.5
                    
                    # Get context window
                    context_start = max(0, position - 10)
                    context_end = min(len(text), position + 10)
                    context_window = text[context_start:context_end]
                    
                    error = NeuralError(
                        position=position,
                        character=character,
                        error_type='high_perplexity',
                        perplexity_score=perplexity,
                        confidence=confidence,
                        suggested_corrections=corrections,
                        context_window=context_window,
                        neural_source='character_lstm'
                    )
                    
                    detected_errors.append(error)
        
        # Determine validity based on perplexity and errors
        perplexity_threshold = 50.0  # Configurable threshold
        is_valid = (overall_perplexity < perplexity_threshold and 
                   len(detected_errors) == 0)
        
        # Calculate model confidence
        model_confidence = min(1.0, 1.0 / (1.0 + overall_perplexity / 10.0))
        
        processing_time = time.time() - start_time
        
        result = NeuralValidationResult(
            text=text,
            is_valid=is_valid,
            overall_perplexity=overall_perplexity,
            character_perplexities=character_perplexities,
            detected_errors=detected_errors,
            processing_time=processing_time,
            model_confidence=model_confidence,
            text_complexity=complexity
        )
        
        return result
    
    def batch_validate(self, 
                      texts: List[str],
                      detailed_analysis: bool = True) -> List[NeuralValidationResult]:
        """
        Validate multiple texts in batch
        
        Args:
            texts: List of texts to validate
            detailed_analysis: Whether to perform detailed analysis
            
        Returns:
            List of validation results
        """
        self.logger.info(f"Batch validating {len(texts)} texts...")
        
        results = []
        start_time = time.time()
        
        for i, text in enumerate(texts):
            result = self.validate_text(text, detailed_analysis)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                self.logger.debug(f"Validated {i + 1}/{len(texts)} texts")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(texts) if texts else 0
        
        self.logger.info(f"Batch validation completed in {total_time:.2f}s")
        self.logger.info(f"Average time per text: {avg_time:.4f}s")
        
        return results
    
    def get_validation_statistics(self, 
                                results: List[NeuralValidationResult]) -> Dict:
        """
        Calculate comprehensive validation statistics
        
        Args:
            results: List of validation results
            
        Returns:
            Dictionary with validation statistics
        """
        if not results:
            return {'error': 'No results provided'}
        
        # Basic stats
        total_texts = len(results)
        valid_texts = sum(1 for r in results if r.is_valid)
        invalid_texts = total_texts - valid_texts
        
        # Perplexity stats
        perplexities = [r.overall_perplexity for r in results]
        avg_perplexity = statistics.mean(perplexities)
        median_perplexity = statistics.median(perplexities)
        
        # Error stats
        total_errors = sum(len(r.detected_errors) for r in results)
        texts_with_errors = sum(1 for r in results if r.detected_errors)
        
        # Processing time stats
        processing_times = [r.processing_time for r in results]
        total_processing_time = sum(processing_times)
        avg_processing_time = statistics.mean(processing_times)
        
        # Error type distribution
        error_types = Counter()
        for result in results:
            for error in result.detected_errors:
                error_types[error.error_type] += 1
        
        # Model confidence stats
        confidences = [r.model_confidence for r in results]
        avg_confidence = statistics.mean(confidences)
        
        return {
            'total_texts': total_texts,
            'valid_texts': valid_texts,
            'invalid_texts': invalid_texts,
            'validation_rate': valid_texts / total_texts,
            'total_errors': total_errors,
            'texts_with_errors': texts_with_errors,
            'avg_errors_per_text': total_errors / total_texts,
            'error_types': dict(error_types),
            'perplexity_stats': {
                'mean': avg_perplexity,
                'median': median_perplexity,
                'min': min(perplexities),
                'max': max(perplexities),
                'std': statistics.stdev(perplexities) if len(perplexities) > 1 else 0
            },
            'performance_stats': {
                'total_processing_time': total_processing_time,
                'avg_processing_time': avg_processing_time,
                'texts_per_second': total_texts / total_processing_time if total_processing_time > 0 else 0
            },
            'confidence_stats': {
                'avg_confidence': avg_confidence,
                'min_confidence': min(confidences),
                'max_confidence': max(confidences)
            }
        }
    
    def configure_thresholds(self, 
                           anomaly_threshold: float = 2.0,
                           confidence_threshold: float = 0.1,
                           context_window: int = 15):
        """
        Configure validation thresholds
        
        Args:
            anomaly_threshold: Multiplier for anomaly detection
            confidence_threshold: Minimum confidence for corrections
            context_window: Size of context window for predictions
        """
        self.anomaly_threshold_multiplier = anomaly_threshold
        self.min_confidence_threshold = confidence_threshold
        self.context_window_size = context_window
        
        self.logger.info(f"Thresholds updated: anomaly={anomaly_threshold}, "
                        f"confidence={confidence_threshold}, context={context_window}")


if __name__ == "__main__":
    # Demo usage
    print("🧠 NEURAL VALIDATOR DEMO")
    print("=" * 28)
    
    # This demo would require a trained model
    # For demonstration, we'll show the structure
    
    print("Note: This demo requires a trained CharacterLSTMModel")
    print("      Run the neural trainer first to create a model")
    
    # Sample validation workflow
    sample_texts = [
        "នេះជាអត្ថបទត្រឹមត្រូវ។",           # Correct text
        "នេះជាអត្ថបទមានកំហុស។",            # Text with potential errors
        "របាយការណ៍ពិសេសអំពីការអភិវឌ្ឍន៍។",    # Complex text
    ]
    
    print(f"\nSample texts for validation:")
    for i, text in enumerate(sample_texts, 1):
        print(f"  {i}. '{text}'")
    
    print(f"\n🔧 Neural validation features:")
    print(f"   ✅ Character-level perplexity scoring")
    print(f"   ✅ Anomaly detection with statistical thresholds")
    print(f"   ✅ Context-aware correction suggestions")
    print(f"   ✅ Batch processing capabilities")
    print(f"   ✅ Comprehensive validation statistics")
    print(f"   ✅ Configurable thresholds and parameters")
    
    print(f"\n🎯 Integration points:")
    print(f"   • Statistical ensemble (Phase 2.4)")
    print(f"   • Rule-based validation")
    print(f"   • N-gram model scoring")
    print(f"   • Real-time spellchecking")
    
    print("\n✅ Neural validator demo structure completed!") 