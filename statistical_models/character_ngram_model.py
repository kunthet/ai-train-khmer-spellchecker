"""
Character N-gram Models for Khmer Spellchecking

This module provides character-level n-gram language models for detecting
spelling errors and unusual character sequences in Khmer text.
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


@dataclass
class ErrorDetectionResult:
    """Result of character-level error detection"""
    text: str
    errors_detected: List[Tuple[int, int, str, float]]  # (start, end, ngram, confidence)
    overall_score: float
    average_probability: float
    min_probability: float
    suspicious_ngrams: List[Tuple[str, float]]
    processing_time: float


@dataclass
class CharacterModelStatistics:
    """Statistics for character n-gram model"""
    n: int
    vocabulary_size: int
    total_ngrams: int
    unique_ngrams: int
    coverage: float
    entropy: float
    smoothing_method: str
    top_ngrams: List[Tuple[str, int]]
    rare_ngrams: List[Tuple[str, int]]
    perplexity: float


class CharacterNgramModel:
    """
    Character-level N-gram language model for Khmer spellchecking
    
    Supports 3-gram, 4-gram, and 5-gram models with various smoothing techniques.
    """
    
    def __init__(
        self,
        n: int,
        smoothing_method: SmoothingMethod = SmoothingMethod.LAPLACE,
        error_threshold: float = 0.001,
        context_window: int = 2,
        filter_non_khmer: bool = True,
        keep_khmer_punctuation: bool = True,
        keep_spaces: bool = True,
        min_khmer_ratio: float = 0.0,
        **smoothing_kwargs
    ):
        """
        Initialize character n-gram model
        
        Args:
            n: N-gram size (3, 4, or 5)
            smoothing_method: Smoothing technique to use
            error_threshold: Probability threshold for error detection
            context_window: Context window for error analysis
            filter_non_khmer: Whether to filter non-Khmer characters
            keep_khmer_punctuation: Keep Khmer punctuation marks (។៕៖ៗ)
            keep_spaces: Keep space characters for context
            min_khmer_ratio: Minimum ratio of Khmer characters to keep text
            **smoothing_kwargs: Additional arguments for smoothing
        """
        if n < 2 or n > 6:
            raise ValueError("N-gram size must be between 2 and 6")
        
        self.n = n
        self.smoothing_method = smoothing_method
        self.error_threshold = error_threshold
        self.context_window = context_window
        
        # Character filtering options
        self.filter_non_khmer = filter_non_khmer
        self.keep_khmer_punctuation = keep_khmer_punctuation
        self.keep_spaces = keep_spaces
        self.min_khmer_ratio = min_khmer_ratio
        
        # Model data
        self.ngram_counts = Counter()
        self.context_counts = Counter()  # (n-1)-gram counts
        self.vocabulary = set()
        self.total_ngrams = 0
        
        # Character filtering statistics
        self.char_filter_stats = {
            'total_chars': 0,
            'khmer_chars': 0,
            'khmer_punct': 0,
            'spaces': 0,
            'filtered_chars': 0,
            'texts_filtered': 0,
            'texts_kept': 0
        }
        
        # Smoothing
        self.smoothing = None
        self.smoothing_kwargs = smoothing_kwargs
        
        # State
        self.is_trained = False
        
        self.logger = logging.getLogger(f"character_ngram_model.{n}gram")
        self.logger.info(f"Initialized {n}-gram model with {smoothing_method.value} smoothing")
        if filter_non_khmer:
            self.logger.info(f"Character filtering enabled: khmer_punct={keep_khmer_punctuation}, "
                           f"spaces={keep_spaces}, min_ratio={min_khmer_ratio}")
    
    def _is_khmer_char(self, char: str) -> bool:
        """Check if character is in Khmer Unicode range"""
        return '\u1780' <= char <= '\u17FF'
    
    def _is_khmer_punctuation(self, char: str) -> bool:
        """Check if character is Khmer punctuation (។៕៖ៗ៙៚ etc.)"""
        return char in '។៕៖ៗ៘៙៚។'
    
    def _should_keep_character(self, char: str) -> str:
        """Determine if character should be kept based on filtering rules"""
        self.char_filter_stats['total_chars'] += 1
        
        # Always keep Khmer characters
        if self._is_khmer_char(char):
            self.char_filter_stats['khmer_chars'] += 1
            return 'khmer'
        
        # Khmer punctuation
        if self.keep_khmer_punctuation and self._is_khmer_punctuation(char):
            self.char_filter_stats['khmer_punct'] += 1
            return 'khmer_punct'
        
        # Spaces
        if self.keep_spaces and char.isspace():
            self.char_filter_stats['spaces'] += 1
            return 'space'
        
        # Non-Khmer character
        self.char_filter_stats['filtered_chars'] += 1
        return 'filtered'
    
    def _filter_text(self, text: str) -> str:
        """Filter text based on character filtering rules"""
        if not self.filter_non_khmer:
            return text
        
        # Calculate Khmer ratio for the entire text
        khmer_count = sum(1 for c in text if self._is_khmer_char(c))
        total_chars = len([c for c in text if not c.isspace()])
        
        if total_chars == 0:
            return ""
        
        khmer_ratio = khmer_count / total_chars
        
        # Filter entire text if Khmer ratio is too low
        if khmer_ratio < self.min_khmer_ratio:
            self.char_filter_stats['texts_filtered'] += 1
            return ""
        
        self.char_filter_stats['texts_kept'] += 1
        
        # Filter individual characters
        filtered_chars = []
        for char in text:
            char_type = self._should_keep_character(char)
            if char_type != 'filtered':
                filtered_chars.append(char)
            # For filtered characters, we skip them entirely
        
        return ''.join(filtered_chars)
    
    def _extract_characters(self, text: str) -> List[str]:
        """Extract characters from text with padding and optional filtering"""
        if not text:
            return []
        
        # Apply filtering if enabled
        if self.filter_non_khmer:
            text = self._filter_text(text)
            if not text:  # Text was completely filtered out
                return []
        
        # Add padding for n-gram extraction
        padding = '<' * (self.n - 1)
        padded_text = padding + text + padding
        
        return list(padded_text)
    
    def _extract_ngrams(self, characters: List[str]) -> List[str]:
        """Extract n-grams from character sequence"""
        if len(characters) < self.n:
            return []
        
        ngrams = []
        for i in range(len(characters) - self.n + 1):
            ngram = ''.join(characters[i:i + self.n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _extract_contexts(self, characters: List[str]) -> List[str]:
        """Extract (n-1)-gram contexts"""
        if len(characters) < self.n - 1:
            return []
        
        contexts = []
        for i in range(len(characters) - self.n + 2):
            context = ''.join(characters[i:i + self.n - 1])
            contexts.append(context)
        
        return contexts
    
    def train_on_texts(self, texts: List[str], show_progress: bool = True) -> CharacterModelStatistics:
        """
        Train the n-gram model on a corpus of texts
        
        Args:
            texts: List of training texts
            show_progress: Whether to show training progress
            
        Returns:
            Model statistics
        """
        start_time = time.time()
        self.logger.info(f"Training {self.n}-gram model on {len(texts)} texts")
        
        # Reset counters
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.vocabulary.clear()
        
        # Process texts
        processed_texts = 0
        for i, text in enumerate(texts):
            if show_progress and i % 1000 == 0:
                self.logger.info(f"Processed {i}/{len(texts)} texts")
            
            # Extract characters and n-grams
            characters = self._extract_characters(text)
            if len(characters) < self.n:
                continue
            
            # Update vocabulary
            self.vocabulary.update(characters)
            
            # Extract and count n-grams
            ngrams = self._extract_ngrams(characters)
            self.ngram_counts.update(ngrams)
            
            # Extract and count contexts
            contexts = self._extract_contexts(characters)
            self.context_counts.update(contexts)
            
            processed_texts += 1
        
        self.total_ngrams = sum(self.ngram_counts.values())
        
        # Initialize smoothing
        vocabulary_size = len(self.vocabulary)
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
        self.logger.info(f"Processed {processed_texts} texts, {self.total_ngrams:,} n-grams")
        self.logger.info(f"Vocabulary size: {vocabulary_size:,} characters")
        
        return self._generate_statistics()
    
    def get_ngram_probability(self, ngram: str) -> float:
        """
        Get probability of an n-gram
        
        Args:
            ngram: N-gram string
            
        Returns:
            Probability of the n-gram
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before computing probabilities")
        
        if len(ngram) != self.n:
            raise ValueError(f"N-gram must be {self.n} characters long")
        
        # Get counts
        ngram_count = self.ngram_counts.get(ngram, 0)
        context = ngram[:-1]
        context_count = self.context_counts.get(context, 0)
        
        # Use smoothing
        if context_count > 0:
            probability = self.smoothing.smooth_probability(ngram_count, context_count)
        else:
            probability = self.smoothing.unseen_probability(self.total_ngrams)
        
        return probability
    
    def get_text_probability(self, text: str) -> float:
        """
        Get overall probability of text using n-gram model
        
        Args:
            text: Text to evaluate
            
        Returns:
            Log probability of the text
        """
        if not text:
            return 0.0
        
        characters = self._extract_characters(text)
        ngrams = self._extract_ngrams(characters)
        
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
    
    def detect_errors(self, text: str) -> ErrorDetectionResult:
        """
        Detect potential spelling errors in text
        
        Args:
            text: Text to check for errors
            
        Returns:
            Error detection result
        """
        start_time = time.time()
        
        if not self.is_trained:
            raise ValueError("Model must be trained before error detection")
        
        characters = self._extract_characters(text)
        ngrams = self._extract_ngrams(characters)
        
        errors_detected = []
        probabilities = []
        suspicious_ngrams = []
        
        # Analyze each n-gram
        for i, ngram in enumerate(ngrams):
            probability = self.get_ngram_probability(ngram)
            probabilities.append(probability)
            
            # Check if probability is below threshold
            if probability < self.error_threshold:
                # Calculate position in original text
                start_pos = max(0, i - (self.n - 1))
                end_pos = min(len(text), start_pos + self.n)
                
                confidence = 1.0 - (probability / self.error_threshold)
                errors_detected.append((start_pos, end_pos, ngram, confidence))
                suspicious_ngrams.append((ngram, probability))
        
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
        
        return ErrorDetectionResult(
            text=text,
            errors_detected=errors_detected,
            overall_score=overall_score,
            average_probability=average_prob,
            min_probability=min_prob,
            suspicious_ngrams=suspicious_ngrams[:10],  # Top 10
            processing_time=processing_time
        )
    
    def _generate_statistics(self) -> CharacterModelStatistics:
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
        coverage = len(self.ngram_counts) / (len(self.vocabulary) ** self.n) if self.vocabulary else 0
        
        # Get top and rare n-grams
        top_ngrams = self.ngram_counts.most_common(20)
        rare_ngrams = [(ngram, count) for ngram, count in self.ngram_counts.items() if count <= 2][:20]
        
        # Add filtering statistics to smoothing method name
        smoothing_info = self.smoothing_method.value
        if self.filter_non_khmer:
            filter_ratio = self.char_filter_stats['filtered_chars'] / max(1, self.char_filter_stats['total_chars'])
            smoothing_info += f" (filtered: {filter_ratio:.1%})"
        
        return CharacterModelStatistics(
            n=self.n,
            vocabulary_size=len(self.vocabulary),
            total_ngrams=self.total_ngrams,
            unique_ngrams=len(self.ngram_counts),
            coverage=coverage,
            entropy=entropy,
            smoothing_method=smoothing_info,
            top_ngrams=top_ngrams,
            rare_ngrams=rare_ngrams,
            perplexity=perplexity
        )
    
    def get_filtering_statistics(self) -> Dict[str, any]:
        """Get character filtering statistics"""
        stats = self.char_filter_stats.copy()
        
        if stats['total_chars'] > 0:
            stats['khmer_ratio'] = stats['khmer_chars'] / stats['total_chars']
            stats['filter_ratio'] = stats['filtered_chars'] / stats['total_chars']
            stats['space_ratio'] = stats['spaces'] / stats['total_chars']
            stats['punct_ratio'] = stats['khmer_punct'] / stats['total_chars']
        else:
            stats.update({
                'khmer_ratio': 0.0,
                'filter_ratio': 0.0,
                'space_ratio': 0.0,
                'punct_ratio': 0.0
            })
        
        total_texts = stats['texts_filtered'] + stats['texts_kept']
        if total_texts > 0:
            stats['text_retention_ratio'] = stats['texts_kept'] / total_texts
        else:
            stats['text_retention_ratio'] = 0.0
        
        return stats
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'n': self.n,
            'smoothing_method': self.smoothing_method.value,
            'error_threshold': self.error_threshold,
            'context_window': self.context_window,
            'filter_non_khmer': self.filter_non_khmer,
            'keep_khmer_punctuation': self.keep_khmer_punctuation,
            'keep_spaces': self.keep_spaces,
            'min_khmer_ratio': self.min_khmer_ratio,
            'smoothing_kwargs': self.smoothing_kwargs,
            'ngram_counts': dict(self.ngram_counts),
            'context_counts': dict(self.context_counts),
            'vocabulary': list(self.vocabulary),
            'total_ngrams': self.total_ngrams,
            'char_filter_stats': self.char_filter_stats,
            'statistics': asdict(self._generate_statistics())
        }
        
        # Save as JSON and pickle
        json_path = f"{filepath}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        pickle_path = f"{filepath}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {json_path} and {pickle_path}")
    
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
        self.error_threshold = model_data['error_threshold']
        self.context_window = model_data['context_window']
        self.smoothing_kwargs = model_data['smoothing_kwargs']
        
        # Restore filtering configuration (with backward compatibility)
        self.filter_non_khmer = model_data.get('filter_non_khmer', False)
        self.keep_khmer_punctuation = model_data.get('keep_khmer_punctuation', True)
        self.keep_spaces = model_data.get('keep_spaces', True)
        self.min_khmer_ratio = model_data.get('min_khmer_ratio', 0.0)
        
        # Restore model data
        self.ngram_counts = Counter(model_data['ngram_counts'])
        self.context_counts = Counter(model_data['context_counts'])
        self.vocabulary = set(model_data['vocabulary'])
        self.total_ngrams = model_data['total_ngrams']
        
        # Restore filtering statistics (with backward compatibility)
        self.char_filter_stats = model_data.get('char_filter_stats', {
            'total_chars': 0, 'khmer_chars': 0, 'khmer_punct': 0,
            'spaces': 0, 'filtered_chars': 0, 'texts_filtered': 0, 'texts_kept': 0
        })
        
        # Reinitialize smoothing
        self.smoothing = create_smoothing_technique(
            self.smoothing_method,
            len(self.vocabulary),
            **self.smoothing_kwargs
        )
        
        # Train Good-Turing if needed
        if self.smoothing_method == SmoothingMethod.GOOD_TURING:
            self.smoothing.train(self.ngram_counts)
        
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")


class NgramModelTrainer:
    """
    Trainer for multiple character n-gram models
    
    Supports training 3-gram, 4-gram, and 5-gram models simultaneously.
    """
    
    def __init__(
        self,
        ngram_sizes: List[int] = [3, 4, 5],
        smoothing_method: SmoothingMethod = SmoothingMethod.LAPLACE,
        filter_non_khmer: bool = True,
        keep_khmer_punctuation: bool = True,
        keep_spaces: bool = True,
        min_khmer_ratio: float = 0.0,
        **smoothing_kwargs
    ):
        """
        Initialize n-gram model trainer
        
        Args:
            ngram_sizes: List of n-gram sizes to train
            smoothing_method: Smoothing method for all models
            filter_non_khmer: Whether to filter non-Khmer characters
            keep_khmer_punctuation: Keep Khmer punctuation marks (។៕៖ៗ)
            keep_spaces: Keep space characters for context
            min_khmer_ratio: Minimum ratio of Khmer characters to keep text
            **smoothing_kwargs: Additional smoothing arguments
        """
        self.ngram_sizes = ngram_sizes
        self.smoothing_method = smoothing_method
        self.smoothing_kwargs = smoothing_kwargs
        
        # Character filtering options
        self.filter_non_khmer = filter_non_khmer
        self.keep_khmer_punctuation = keep_khmer_punctuation
        self.keep_spaces = keep_spaces
        self.min_khmer_ratio = min_khmer_ratio
        
        self.models = {}
        self.training_stats = {}
        
        self.logger = logging.getLogger("ngram_model_trainer")
        
        # Initialize models
        for n in ngram_sizes:
            self.models[n] = CharacterNgramModel(
                n=n,
                smoothing_method=smoothing_method,
                filter_non_khmer=filter_non_khmer,
                keep_khmer_punctuation=keep_khmer_punctuation,
                keep_spaces=keep_spaces,
                min_khmer_ratio=min_khmer_ratio,
                **smoothing_kwargs
            )
    
    def train_all_models(self, texts: List[str], show_progress: bool = True) -> Dict[int, CharacterModelStatistics]:
        """
        Train all n-gram models
        
        Args:
            texts: Training texts
            show_progress: Whether to show progress
            
        Returns:
            Dictionary of n-gram sizes to statistics
        """
        self.logger.info(f"Training {len(self.ngram_sizes)} n-gram models on {len(texts)} texts")
        
        results = {}
        for n in self.ngram_sizes:
            self.logger.info(f"Training {n}-gram model...")
            stats = self.models[n].train_on_texts(texts, show_progress)
            results[n] = stats
            self.training_stats[n] = stats
        
        self.logger.info("All models trained successfully")
        return results
    
    def get_model(self, n: int) -> CharacterNgramModel:
        """Get specific n-gram model"""
        if n not in self.models:
            raise ValueError(f"Model for {n}-grams not available")
        return self.models[n]
    
    def detect_errors_ensemble(self, text: str) -> Dict[int, ErrorDetectionResult]:
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
                filepath = output_path / f"char_{n}gram_model"
                model.save_model(str(filepath))
        
        self.logger.info(f"All models saved to {output_dir}")
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        report = []
        report.append("📊 CHARACTER N-GRAM MODELS TRAINING REPORT")
        report.append("=" * 55)
        report.append("")
        
        # Overview
        report.append("📋 OVERVIEW")
        report.append("-" * 15)
        report.append(f"Models trained: {len(self.training_stats)}")
        report.append(f"N-gram sizes: {', '.join(map(str, sorted(self.training_stats.keys())))}")
        report.append(f"Smoothing method: {self.smoothing_method.value}")
        report.append("")
        
        # Model details
        for n in sorted(self.training_stats.keys()):
            stats = self.training_stats[n]
            report.append(f"🔢 {n}-GRAM MODEL")
            report.append("-" * 20)
            report.append(f"Vocabulary size: {stats.vocabulary_size:,}")
            report.append(f"Total n-grams: {stats.total_ngrams:,}")
            report.append(f"Unique n-grams: {stats.unique_ngrams:,}")
            report.append(f"Coverage: {stats.coverage:.6f}")
            report.append(f"Entropy: {stats.entropy:.3f}")
            report.append(f"Perplexity: {stats.perplexity:.1f}")
            
            # Top n-grams
            report.append(f"Top 5 {n}-grams:")
            for i, (ngram, count) in enumerate(stats.top_ngrams[:5], 1):
                pct = (count / stats.total_ngrams) * 100
                report.append(f"  {i}. '{ngram}': {count:,} ({pct:.3f}%)")
            
            report.append("")
        
        return "\n".join(report)
    
    def generate_filtering_report(self) -> str:
        """Generate comprehensive filtering statistics report"""
        if not self.training_stats:
            return "No training statistics available"
        
        report = []
        report.append("📋 CHARACTER FILTERING STATISTICS")
        report.append("=" * 40)
        report.append("")
        
        # Overall filtering configuration
        report.append("🔧 FILTERING CONFIGURATION")
        report.append("-" * 25)
        report.append(f"Filter non-Khmer: {self.filter_non_khmer}")
        report.append(f"Keep Khmer punctuation: {self.keep_khmer_punctuation}")
        report.append(f"Keep spaces: {self.keep_spaces}")
        report.append(f"Min Khmer ratio: {self.min_khmer_ratio}")
        report.append("")
        
        # Per-model statistics
        for n in sorted(self.training_stats.keys()):
            model = self.models[n]
            filter_stats = model.get_filtering_statistics()
            
            report.append(f"📊 {n}-GRAM MODEL FILTERING")
            report.append("-" * 25)
            report.append(f"Total characters processed: {filter_stats['total_chars']:,}")
            report.append(f"Khmer characters: {filter_stats['khmer_chars']:,} ({filter_stats['khmer_ratio']:.1%})")
            report.append(f"Khmer punctuation: {filter_stats['khmer_punct']:,} ({filter_stats['punct_ratio']:.1%})")
            report.append(f"Spaces: {filter_stats['spaces']:,} ({filter_stats['space_ratio']:.1%})")
            report.append(f"Filtered out: {filter_stats['filtered_chars']:,} ({filter_stats['filter_ratio']:.1%})")
            report.append("")
            report.append(f"Texts processed: {filter_stats['texts_kept'] + filter_stats['texts_filtered']:,}")
            report.append(f"Texts kept: {filter_stats['texts_kept']:,}")
            report.append(f"Texts filtered: {filter_stats['texts_filtered']:,}")
            report.append(f"Text retention rate: {filter_stats['text_retention_ratio']:.1%}")
            report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Demo usage
    print("📊 CHARACTER N-GRAM MODEL DEMO")
    print("=" * 40)
    
    # Sample training texts
    sample_texts = [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
        "ខ្ញុំស្រលាញ់ភាសាខ្មែរណាស់។",
        "ប្រទេសកម្ពុជាមានទីរដ្ឋធានីភ្នំពេញ។",
        "កម្មវិធីនេះមានប្រយោជន៍ច្រើន។"
    ]
    
    # Train models
    trainer = NgramModelTrainer()
    stats = trainer.train_all_models(sample_texts, show_progress=False)
    
    print(trainer.generate_training_report())
    
    # Test error detection
    test_text = "នេះជាការសាកល្បងខុស"
    print("🔍 ERROR DETECTION TEST")
    print(f"Text: {test_text}")
    
    results = trainer.detect_errors_ensemble(test_text)
    for n, result in results.items():
        print(f"\n{n}-gram results:")
        print(f"  Overall score: {result.overall_score:.3f}")
        print(f"  Errors detected: {len(result.errors_detected)}")
        print(f"  Average probability: {result.average_probability:.6f}") 