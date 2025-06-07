"""
Syllable N-gram Models for Khmer Spellchecking

This module provides syllable-level n-gram language models for detecting
spelling errors and unusual syllable sequences in Khmer text.
"""

import math
import logging
import json
import pickle
import time
from typing import Dict, List, Tuple, Optional, Counter, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from .smoothing_techniques import (
        BaseSmoothingTechnique, 
        create_smoothing_technique, 
        SmoothingMethod
    )
except ImportError:
    # Fallback for running as standalone script
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from smoothing_techniques import (
        BaseSmoothingTechnique, 
        create_smoothing_technique, 
        SmoothingMethod
    )

# Import syllable segmentation
try:
    from word_cluster.syllable_api import SyllableSegmentationAPI, SegmentationMethod
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from word_cluster.syllable_api import SyllableSegmentationAPI, SegmentationMethod


@dataclass
class SyllableErrorDetectionResult:
    """Result of syllable-level error detection"""
    text: str
    syllables: List[str]
    errors_detected: List[Tuple[int, int, str, float]]  # (start, end, syllable, confidence)
    overall_score: float
    average_probability: float
    min_probability: float
    suspicious_syllables: List[Tuple[str, float]]
    invalid_sequences: List[Tuple[str, float]]  # Invalid syllable combinations
    processing_time: float


@dataclass
class SyllableModelStatistics:
    """Statistics for syllable n-gram model"""
    n: int
    vocabulary_size: int
    total_ngrams: int
    unique_ngrams: int
    coverage: float
    entropy: float
    smoothing_method: str
    segmentation_method: str
    top_ngrams: List[Tuple[str, int]]
    rare_ngrams: List[Tuple[str, int]]
    perplexity: float
    avg_syllable_length: float
    syllable_diversity: float


class SyllableNgramModel:
    """
    Syllable-level N-gram language model for Khmer spellchecking
    
    Supports 2-gram, 3-gram, and 4-gram models with syllable-aware processing.
    """
    
    def __init__(
        self,
        n: int,
        smoothing_method: SmoothingMethod = SmoothingMethod.LAPLACE,
        segmentation_method: SegmentationMethod = SegmentationMethod.REGEX_ADVANCED,
        error_threshold: float = 0.001,
        context_window: int = 2,
        filter_invalid_syllables: bool = True,
        min_syllable_length: int = 1,
        max_syllable_length: int = 10,
        **smoothing_kwargs
    ):
        """
        Initialize syllable n-gram model
        
        Args:
            n: N-gram size (2, 3, or 4)
            smoothing_method: Smoothing technique to use
            segmentation_method: Syllable segmentation method
            error_threshold: Probability threshold for error detection
            context_window: Context window for error analysis
            filter_invalid_syllables: Whether to filter invalid syllables
            min_syllable_length: Minimum syllable length
            max_syllable_length: Maximum syllable length
            **smoothing_kwargs: Additional arguments for smoothing
        """
        if n < 2 or n > 5:
            raise ValueError("N-gram size must be between 2 and 5")
        
        self.n = n
        self.smoothing_method = smoothing_method
        self.segmentation_method = segmentation_method
        self.error_threshold = error_threshold
        self.context_window = context_window
        
        # Syllable filtering options
        self.filter_invalid_syllables = filter_invalid_syllables
        self.min_syllable_length = min_syllable_length
        self.max_syllable_length = max_syllable_length
        
        # Model data
        self.ngram_counts = Counter()
        self.context_counts = Counter()  # (n-1)-gram counts
        self.syllable_vocabulary = set()
        self.total_ngrams = 0
        
        # Syllable statistics
        self.syllable_frequencies = Counter()
        self.syllable_lengths = Counter()
        self.invalid_syllables = set()
        
        # Segmentation API
        self.segmentation_api = SyllableSegmentationAPI(segmentation_method)
        
        # Smoothing
        self.smoothing = None
        self.smoothing_kwargs = smoothing_kwargs
        
        # State
        self.is_trained = False
        
        self.logger = logging.getLogger(f"syllable_ngram_model.{n}gram")
        self.logger.info(f"Initialized {n}-gram syllable model with {smoothing_method.value} smoothing")
        self.logger.info(f"Using {segmentation_method.value} segmentation method")
    
    def _is_valid_syllable(self, syllable: str) -> bool:
        """Check if syllable is valid based on filtering criteria"""
        if not syllable or not syllable.strip():
            return False
        
        # Length check
        if len(syllable) < self.min_syllable_length or len(syllable) > self.max_syllable_length:
            return False
        
        # Basic Khmer character check (at least 50% Khmer characters)
        if self.filter_invalid_syllables:
            khmer_chars = sum(1 for c in syllable if '\u1780' <= c <= '\u17FF')
            total_chars = len([c for c in syllable if not c.isspace()])
            
            if total_chars > 0:
                khmer_ratio = khmer_chars / total_chars
                if khmer_ratio < 0.5:  # At least 50% Khmer
                    return False
        
        return True
    
    def _extract_syllables(self, text: str) -> List[str]:
        """Extract syllables from text using configured segmentation method"""
        try:
            result = self.segmentation_api.segment_text(text)
            if result.success:
                # Filter syllables if enabled
                if self.filter_invalid_syllables:
                    valid_syllables = []
                    for syllable in result.syllables:
                        if self._is_valid_syllable(syllable):
                            valid_syllables.append(syllable)
                        else:
                            self.invalid_syllables.add(syllable)
                    return valid_syllables
                else:
                    return result.syllables
            else:
                self.logger.warning(f"Segmentation failed for text: {text[:50]}...")
                return []
        except Exception as e:
            self.logger.error(f"Error segmenting text: {e}")
            return []
    
    def _extract_ngrams(self, syllables: List[str]) -> List[str]:
        """Extract n-grams from syllable sequence with padding"""
        if len(syllables) < self.n:
            return []
        
        # Add padding for n-gram extraction
        padding = ['<START>'] * (self.n - 1)
        padded_syllables = padding + syllables + ['<END>'] * (self.n - 1)
        
        ngrams = []
        for i in range(len(padded_syllables) - self.n + 1):
            ngram = ' | '.join(padded_syllables[i:i + self.n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _extract_contexts(self, syllables: List[str]) -> List[str]:
        """Extract (n-1)-gram contexts with padding"""
        if len(syllables) < self.n - 1:
            return []
        
        # Add padding for context extraction
        padding = ['<START>'] * (self.n - 1)
        padded_syllables = padding + syllables + ['<END>'] * (self.n - 1)
        
        contexts = []
        for i in range(len(padded_syllables) - self.n + 2):
            context = ' | '.join(padded_syllables[i:i + self.n - 1])
            contexts.append(context)
        
        return contexts
    
    def train_on_texts(self, texts: List[str], show_progress: bool = True) -> SyllableModelStatistics:
        """
        Train the syllable n-gram model on a corpus of texts
        
        Args:
            texts: List of training texts
            show_progress: Whether to show training progress
            
        Returns:
            Model statistics
        """
        start_time = time.time()
        self.logger.info(f"Training {self.n}-gram syllable model on {len(texts)} texts")
        
        # Reset counters
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.syllable_vocabulary.clear()
        self.syllable_frequencies.clear()
        self.syllable_lengths.clear()
        self.invalid_syllables.clear()
        
        # Process texts
        processed_texts = 0
        total_syllables = 0
        
        for i, text in enumerate(texts):
            if show_progress and i % 1000 == 0:
                self.logger.info(f"Processed {i}/{len(texts)} texts")
            
            # Extract syllables
            syllables = self._extract_syllables(text)
            if len(syllables) < self.n:
                continue
            
            # Update syllable statistics
            self.syllable_vocabulary.update(syllables)
            self.syllable_frequencies.update(syllables)
            for syllable in syllables:
                self.syllable_lengths[len(syllable)] += 1
            
            # Extract and count n-grams
            ngrams = self._extract_ngrams(syllables)
            self.ngram_counts.update(ngrams)
            
            # Extract and count contexts
            contexts = self._extract_contexts(syllables)
            self.context_counts.update(contexts)
            
            processed_texts += 1
            total_syllables += len(syllables)
        
        self.total_ngrams = sum(self.ngram_counts.values())
        
        # Initialize smoothing
        vocabulary_size = len(self.syllable_vocabulary)
        self.smoothing = create_smoothing_technique(
            self.smoothing_method, 
            vocabulary_size, 
            **self.smoothing_kwargs
        )
        
        # Train Good-Turing if needed
        if self.smoothing_method == SmoothingMethod.GOOD_TURING:
            self.smoothing.train(self.ngram_counts)
        
        self.is_trained = True
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f}s")
        self.logger.info(f"Processed {processed_texts} texts, {total_syllables:,} syllables")
        self.logger.info(f"Generated {self.total_ngrams:,} n-grams")
        self.logger.info(f"Syllable vocabulary size: {vocabulary_size:,}")
        if self.invalid_syllables:
            self.logger.info(f"Filtered {len(self.invalid_syllables):,} invalid syllables")
        
        return self._generate_statistics()
    
    def get_ngram_probability(self, ngram: str) -> float:
        """
        Get probability of a syllable n-gram
        
        Args:
            ngram: N-gram string (syllables separated by ' | ')
            
        Returns:
            Probability of the n-gram
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before computing probabilities")
        
        syllables = ngram.split(' | ')
        if len(syllables) != self.n:
            raise ValueError(f"N-gram must contain {self.n} syllables")
        
        # Get counts
        ngram_count = self.ngram_counts.get(ngram, 0)
        context = ' | '.join(syllables[:-1])
        context_count = self.context_counts.get(context, 0)
        
        # Use smoothing
        if context_count > 0:
            probability = self.smoothing.smooth_probability(ngram_count, context_count)
        else:
            probability = self.smoothing.unseen_probability(self.total_ngrams)
        
        return probability
    
    def get_text_probability(self, text: str) -> float:
        """
        Get overall probability of text using syllable n-gram model
        
        Args:
            text: Text to evaluate
            
        Returns:
            Log probability of the text
        """
        if not text:
            return 0.0
        
        syllables = self._extract_syllables(text)
        ngrams = self._extract_ngrams(syllables)
        
        if not ngrams:
            return float('-inf')
        
        log_prob = 0.0
        for ngram in ngrams:
            prob = self.get_ngram_probability(ngram)
            if prob > 0:
                log_prob += math.log(prob)
            else:
                log_prob += math.log(1e-10)  # Small value to avoid log(0)
        
        # Average log probability
        return log_prob / len(ngrams)
    
    def detect_errors(self, text: str) -> SyllableErrorDetectionResult:
        """
        Detect potential spelling errors in text at syllable level
        
        Args:
            text: Text to check for errors
            
        Returns:
            Syllable error detection result
        """
        start_time = time.time()
        
        if not self.is_trained:
            raise ValueError("Model must be trained before error detection")
        
        syllables = self._extract_syllables(text)
        ngrams = self._extract_ngrams(syllables)
        
        errors_detected = []
        probabilities = []
        suspicious_syllables = []
        invalid_sequences = []
        
        # Analyze each n-gram
        syllable_positions = {}
        current_pos = 0
        for i, syllable in enumerate(syllables):
            syllable_positions[i] = (current_pos, current_pos + len(syllable))
            current_pos += len(syllable)
        
        for i, ngram in enumerate(ngrams):
            probability = self.get_ngram_probability(ngram)
            probabilities.append(probability)
            
            # Check if probability is below threshold
            if probability < self.error_threshold:
                # Find the problematic syllable (usually the last one in the n-gram)
                ngram_syllables = ngram.split(' | ')
                target_syllable_idx = min(i + self.n - 1, len(syllables) - 1)
                
                if target_syllable_idx in syllable_positions:
                    start_pos, end_pos = syllable_positions[target_syllable_idx]
                    target_syllable = syllables[target_syllable_idx]
                    
                    confidence = 1.0 - (probability / self.error_threshold)
                    errors_detected.append((start_pos, end_pos, target_syllable, confidence))
                    
                    # Check if it's a suspicious syllable or invalid sequence
                    if target_syllable not in self.syllable_vocabulary:
                        suspicious_syllables.append((target_syllable, probability))
                    else:
                        invalid_sequences.append((ngram, probability))
        
        # Calculate overall statistics
        if probabilities:
            average_prob = sum(probabilities) / len(probabilities)
            min_prob = min(probabilities)
            overall_score = 1.0 - (len(errors_detected) / len(ngrams))
        else:
            average_prob = 0.0
            min_prob = 0.0
            overall_score = 0.0
        
        processing_time = time.time() - start_time
        
        return SyllableErrorDetectionResult(
            text=text,
            syllables=syllables,
            errors_detected=errors_detected,
            overall_score=overall_score,
            average_probability=average_prob,
            min_probability=min_prob,
            suspicious_syllables=suspicious_syllables[:10],  # Top 10
            invalid_sequences=invalid_sequences[:10],  # Top 10
            processing_time=processing_time
        )
    
    def _generate_statistics(self) -> SyllableModelStatistics:
        """Generate model statistics"""
        if not self.is_trained:
            raise ValueError("Model must be trained to generate statistics")
        
        # Calculate entropy
        entropy = 0.0
        for ngram, count in self.ngram_counts.items():
            prob = count / self.total_ngrams
            entropy -= prob * math.log2(prob)
        
        # Calculate perplexity
        perplexity = 2 ** entropy
        
        # Calculate coverage
        coverage = len(self.ngram_counts) / (len(self.syllable_vocabulary) ** self.n) if self.syllable_vocabulary else 0
        
        # Calculate average syllable length
        if self.syllable_lengths:
            total_chars = sum(length * count for length, count in self.syllable_lengths.items())
            total_syllables = sum(self.syllable_lengths.values())
            avg_syllable_length = total_chars / total_syllables
        else:
            avg_syllable_length = 0.0
        
        # Calculate syllable diversity (entropy of syllable frequencies)
        syllable_diversity = 0.0
        if self.syllable_frequencies:
            total_freq = sum(self.syllable_frequencies.values())
            for freq in self.syllable_frequencies.values():
                prob = freq / total_freq
                syllable_diversity -= prob * math.log2(prob)
        
        # Get top and rare n-grams
        top_ngrams = self.ngram_counts.most_common(20)
        rare_ngrams = [(ngram, count) for ngram, count in self.ngram_counts.items() if count <= 2][:20]
        
        return SyllableModelStatistics(
            n=self.n,
            vocabulary_size=len(self.syllable_vocabulary),
            total_ngrams=self.total_ngrams,
            unique_ngrams=len(self.ngram_counts),
            coverage=coverage,
            entropy=entropy,
            smoothing_method=self.smoothing_method.value,
            segmentation_method=self.segmentation_method.value,
            top_ngrams=top_ngrams,
            rare_ngrams=rare_ngrams,
            perplexity=perplexity,
            avg_syllable_length=avg_syllable_length,
            syllable_diversity=syllable_diversity
        )
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'n': self.n,
            'smoothing_method': self.smoothing_method.value,
            'segmentation_method': self.segmentation_method.value,
            'error_threshold': self.error_threshold,
            'context_window': self.context_window,
            'filter_invalid_syllables': self.filter_invalid_syllables,
            'min_syllable_length': self.min_syllable_length,
            'max_syllable_length': self.max_syllable_length,
            'smoothing_kwargs': self.smoothing_kwargs,
            'ngram_counts': dict(self.ngram_counts),
            'context_counts': dict(self.context_counts),
            'syllable_vocabulary': list(self.syllable_vocabulary),
            'syllable_frequencies': dict(self.syllable_frequencies),
            'syllable_lengths': dict(self.syllable_lengths),
            'invalid_syllables': list(self.invalid_syllables),
            'total_ngrams': self.total_ngrams,
            'statistics': asdict(self._generate_statistics())
        }
        
        # Save as JSON and pickle
        json_path = f"{filepath}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        pickle_path = f"{filepath}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Syllable model saved to {json_path} and {pickle_path}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        # Try pickle first (faster)
        pickle_path = f"{filepath}.pkl"
        if Path(pickle_path).exists():
            with open(pickle_path, 'rb') as f:
                model_data = pickle.load(f)
        else:
            # Fall back to JSON
            json_path = f"{filepath}.json"
            with open(json_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        
        # Restore model state
        self.n = model_data['n']
        self.smoothing_method = SmoothingMethod(model_data['smoothing_method'])
        self.segmentation_method = SegmentationMethod(model_data['segmentation_method'])
        self.error_threshold = model_data['error_threshold']
        self.context_window = model_data['context_window']
        self.filter_invalid_syllables = model_data.get('filter_invalid_syllables', True)
        self.min_syllable_length = model_data.get('min_syllable_length', 1)
        self.max_syllable_length = model_data.get('max_syllable_length', 10)
        self.smoothing_kwargs = model_data['smoothing_kwargs']
        
        # Restore model data
        self.ngram_counts = Counter(model_data['ngram_counts'])
        self.context_counts = Counter(model_data['context_counts'])
        self.syllable_vocabulary = set(model_data['syllable_vocabulary'])
        self.syllable_frequencies = Counter(model_data.get('syllable_frequencies', {}))
        self.syllable_lengths = Counter(model_data.get('syllable_lengths', {}))
        self.invalid_syllables = set(model_data.get('invalid_syllables', []))
        self.total_ngrams = model_data['total_ngrams']
        
        # Reinitialize APIs
        self.segmentation_api = SyllableSegmentationAPI(self.segmentation_method)
        
        # Reinitialize smoothing
        self.smoothing = create_smoothing_technique(
            self.smoothing_method,
            len(self.syllable_vocabulary),
            **self.smoothing_kwargs
        )
        
        # Train Good-Turing if needed
        if self.smoothing_method == SmoothingMethod.GOOD_TURING:
            self.smoothing.train(self.ngram_counts)
        
        self.is_trained = True
        self.logger.info(f"Syllable model loaded from {filepath}")


class SyllableNgramModelTrainer:
    """
    Trainer for multiple syllable n-gram models
    
    Supports training 2-gram, 3-gram, and 4-gram models simultaneously.
    """
    
    def __init__(
        self,
        ngram_sizes: List[int] = [2, 3, 4],
        smoothing_method: SmoothingMethod = SmoothingMethod.LAPLACE,
        segmentation_method: SegmentationMethod = SegmentationMethod.REGEX_ADVANCED,
        **model_kwargs
    ):
        """
        Initialize syllable n-gram model trainer
        
        Args:
            ngram_sizes: List of n-gram sizes to train
            smoothing_method: Smoothing method for all models
            segmentation_method: Syllable segmentation method
            **model_kwargs: Additional model arguments
        """
        self.ngram_sizes = ngram_sizes
        self.smoothing_method = smoothing_method
        self.segmentation_method = segmentation_method
        self.model_kwargs = model_kwargs
        
        self.models = {}
        self.training_stats = {}
        
        self.logger = logging.getLogger("syllable_ngram_model_trainer")
        
        # Initialize models
        for n in ngram_sizes:
            self.models[n] = SyllableNgramModel(
                n=n,
                smoothing_method=smoothing_method,
                segmentation_method=segmentation_method,
                **model_kwargs
            )
    
    def train_all_models(self, texts: List[str], show_progress: bool = True) -> Dict[int, SyllableModelStatistics]:
        """
        Train all syllable n-gram models
        
        Args:
            texts: Training texts
            show_progress: Whether to show progress
            
        Returns:
            Dictionary of n-gram sizes to statistics
        """
        self.logger.info(f"Training {len(self.ngram_sizes)} syllable n-gram models on {len(texts)} texts")
        
        results = {}
        for n in self.ngram_sizes:
            self.logger.info(f"Training {n}-gram syllable model...")
            stats = self.models[n].train_on_texts(texts, show_progress)
            results[n] = stats
            self.training_stats[n] = stats
        
        self.logger.info("All syllable models trained successfully")
        return results
    
    def get_model(self, n: int) -> SyllableNgramModel:
        """Get specific n-gram model"""
        if n not in self.models:
            raise ValueError(f"Model for {n}-grams not available")
        return self.models[n]
    
    def detect_errors_ensemble(self, text: str) -> Dict[int, SyllableErrorDetectionResult]:
        """
        Run error detection with all models
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of n-gram sizes to error detection results
        """
        results = {}
        for n, model in self.models.items():
            if model.is_trained:
                results[n] = model.detect_errors(text)
        return results
    
    def save_all_models(self, output_dir: str):
        """Save all trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for n, model in self.models.items():
            if model.is_trained:
                filepath = output_path / f"syllable_{n}gram_model"
                model.save_model(str(filepath))
        
        self.logger.info(f"All syllable models saved to {output_dir}")
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        report = []
        report.append("📊 SYLLABLE N-GRAM MODELS TRAINING REPORT")
        report.append("=" * 55)
        report.append("")
        
        # Overview
        report.append("📋 OVERVIEW")
        report.append("-" * 15)
        report.append(f"Models trained: {len(self.training_stats)}")
        report.append(f"N-gram sizes: {', '.join(map(str, sorted(self.training_stats.keys())))}")
        report.append(f"Smoothing method: {self.smoothing_method.value}")
        report.append(f"Segmentation method: {self.segmentation_method.value}")
        report.append("")
        
        # Model details
        for n in sorted(self.training_stats.keys()):
            stats = self.training_stats[n]
            report.append(f"🔢 {n}-GRAM SYLLABLE MODEL")
            report.append("-" * 25)
            report.append(f"Syllable vocabulary size: {stats.vocabulary_size:,}")
            report.append(f"Total n-grams: {stats.total_ngrams:,}")
            report.append(f"Unique n-grams: {stats.unique_ngrams:,}")
            report.append(f"Coverage: {stats.coverage:.6f}")
            report.append(f"Entropy: {stats.entropy:.3f}")
            report.append(f"Perplexity: {stats.perplexity:.1f}")
            report.append(f"Avg syllable length: {stats.avg_syllable_length:.2f}")
            report.append(f"Syllable diversity: {stats.syllable_diversity:.3f}")
            
            # Top n-grams
            report.append(f"Top 5 {n}-grams:")
            for i, (ngram, count) in enumerate(stats.top_ngrams[:5], 1):
                pct = (count / stats.total_ngrams) * 100
                report.append(f"  {i}. '{ngram}': {count:,} ({pct:.3f}%)")
            
            report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Demo usage
    print("📊 SYLLABLE N-GRAM MODEL DEMO")
    print("=" * 40)
    
    # Sample training texts
    sample_texts = [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
        "ខ្ញុំស្រលាញ់ភាសាខ្មែរណាស់។",
        "ប្រទេសកម្ពុជាមានទីរដ្ឋធានីភ្នំពេញ។",
        "កម្មវិធីនេះមានប្រយោជន៍ច្រើន។"
    ]
    
    # Train models
    trainer = SyllableNgramModelTrainer()
    stats = trainer.train_all_models(sample_texts, show_progress=False)
    
    print(trainer.generate_training_report())
    
    # Test error detection
    test_text = "នេះជាការសាកល្បងខុស"
    print("🔍 SYLLABLE ERROR DETECTION TEST")
    print(f"Text: {test_text}")
    
    results = trainer.detect_errors_ensemble(test_text)
    for n, result in results.items():
        print(f"\n{n}-gram results:")
        print(f"  Syllables: {result.syllables}")
        print(f"  Overall score: {result.overall_score:.3f}")
        print(f"  Errors detected: {len(result.errors_detected)}")
        print(f"  Suspicious syllables: {len(result.suspicious_syllables)}")
        print(f"  Invalid sequences: {len(result.invalid_sequences)}") 