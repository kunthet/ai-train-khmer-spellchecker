"""
Smoothing Techniques for N-gram Models

This module provides various smoothing techniques to handle unseen n-grams
and improve probability estimation in character-level language models.
"""

import math
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Counter
from collections import Counter, defaultdict
from enum import Enum
from dataclasses import dataclass


class SmoothingMethod(Enum):
    """Available smoothing methods"""
    LAPLACE = "laplace"
    GOOD_TURING = "good_turing"
    SIMPLE_BACKOFF = "simple_backoff"


@dataclass
class SmoothingStatistics:
    """Statistics for smoothing evaluation"""
    method: str
    vocabulary_size: int
    total_ngrams: int
    unseen_probability: float
    smoothing_factor: float
    coverage: float


class BaseSmoothingTechnique(ABC):
    """Abstract base class for smoothing techniques"""
    
    def __init__(self, vocabulary_size: int):
        self.vocabulary_size = vocabulary_size
        self.logger = logging.getLogger(f"smoothing.{self.__class__.__name__}")
    
    @abstractmethod
    def smooth_probability(self, ngram_count: int, total_count: int, context_count: int = None) -> float:
        """Calculate smoothed probability for an n-gram"""
        pass
    
    @abstractmethod
    def unseen_probability(self, total_count: int, context_count: int = None) -> float:
        """Calculate probability for unseen n-grams"""
        pass


class LaplaceSmoothing(BaseSmoothingTechnique):
    """
    Laplace (Add-One) Smoothing
    
    Adds a small count (alpha) to all n-gram counts to handle unseen events.
    Simple but effective for character-level models.
    """
    
    def __init__(self, vocabulary_size: int, alpha: float = 1.0):
        super().__init__(vocabulary_size)
        self.alpha = alpha
        self.logger.info(f"Initialized Laplace smoothing with alpha={alpha}")
    
    def smooth_probability(self, ngram_count: int, total_count: int, context_count: int = None) -> float:
        """
        Calculate Laplace-smoothed probability
        
        P(w|context) = (count(context,w) + alpha) / (count(context) + alpha * V)
        
        Args:
            ngram_count: Count of the specific n-gram
            total_count: Total count of all n-grams with same context
            context_count: Count of the context (not used in Laplace)
            
        Returns:
            Smoothed probability
        """
        numerator = ngram_count + self.alpha
        denominator = total_count + (self.alpha * self.vocabulary_size)
        
        if denominator == 0:
            return 1.0 / self.vocabulary_size
        
        return numerator / denominator
    
    def unseen_probability(self, total_count: int, context_count: int = None) -> float:
        """Calculate probability for unseen n-grams"""
        return self.smooth_probability(0, total_count, context_count)
    
    def get_statistics(self, total_count: int) -> SmoothingStatistics:
        """Get smoothing statistics"""
        unseen_prob = self.unseen_probability(total_count)
        coverage = total_count / (total_count + self.alpha * self.vocabulary_size)
        
        return SmoothingStatistics(
            method="laplace",
            vocabulary_size=self.vocabulary_size,
            total_ngrams=total_count,
            unseen_probability=unseen_prob,
            smoothing_factor=self.alpha,
            coverage=coverage
        )


class GoodTuringSmoothing(BaseSmoothingTechnique):
    """
    Good-Turing Smoothing
    
    Uses frequency of frequencies to estimate probabilities of unseen events.
    More sophisticated than Laplace, better for natural language.
    """
    
    def __init__(self, vocabulary_size: int, use_simple_gt: bool = True):
        super().__init__(vocabulary_size)
        self.use_simple_gt = use_simple_gt
        self.frequency_of_frequencies = Counter()
        self.adjusted_counts = {}
        self.total_mass = 0
        self.is_trained = False
        self.logger.info(f"Initialized Good-Turing smoothing (simple={use_simple_gt})")
    
    def train(self, ngram_counts: Counter):
        """
        Train Good-Turing smoothing on n-gram counts
        
        Args:
            ngram_counts: Counter of n-gram counts
        """
        # Calculate frequency of frequencies
        self.frequency_of_frequencies = Counter(ngram_counts.values())
        self.total_mass = sum(ngram_counts.values())
        
        # Calculate adjusted counts using Good-Turing formula
        self._calculate_adjusted_counts()
        
        self.is_trained = True
        self.logger.info(f"Trained Good-Turing on {len(ngram_counts)} n-grams")
        self.logger.info(f"Frequency of frequencies: {dict(list(self.frequency_of_frequencies.most_common(10)))}")
    
    def _calculate_adjusted_counts(self):
        """Calculate Good-Turing adjusted counts"""
        self.adjusted_counts = {}
        
        # Simple Good-Turing formula: c* = (c+1) * N(c+1) / N(c)
        for count in self.frequency_of_frequencies:
            if count == 0:
                continue
                
            next_count = count + 1
            current_freq = self.frequency_of_frequencies[count]
            next_freq = self.frequency_of_frequencies.get(next_count, 0)
            
            if next_freq > 0:
                adjusted = next_count * next_freq / current_freq
                self.adjusted_counts[count] = max(adjusted, count * 0.1)  # Avoid zero
            else:
                # Use Simple Good-Turing for unseen frequencies
                self.adjusted_counts[count] = count
        
        # Handle unseen n-grams (count = 0)
        if 1 in self.frequency_of_frequencies:
            self.adjusted_counts[0] = self.frequency_of_frequencies[1] / self.total_mass
        else:
            self.adjusted_counts[0] = 1.0 / (self.total_mass + self.vocabulary_size)
    
    def smooth_probability(self, ngram_count: int, total_count: int, context_count: int = None) -> float:
        """Calculate Good-Turing smoothed probability"""
        if not self.is_trained:
            # Fallback to simple uniform distribution
            return 1.0 / self.vocabulary_size
        
        # Get adjusted count
        adjusted_count = self.adjusted_counts.get(ngram_count, ngram_count)
        
        # Calculate probability
        if total_count > 0:
            return adjusted_count / total_count
        else:
            return 1.0 / self.vocabulary_size
    
    def unseen_probability(self, total_count: int, context_count: int = None) -> float:
        """Calculate probability for unseen n-grams"""
        if not self.is_trained:
            return 1.0 / self.vocabulary_size
        
        return self.adjusted_counts.get(0, 1.0 / self.vocabulary_size)
    
    def get_statistics(self, total_count: int) -> SmoothingStatistics:
        """Get smoothing statistics"""
        unseen_prob = self.unseen_probability(total_count)
        
        # Calculate coverage (proportion of mass assigned to seen events)
        seen_mass = 1.0 - (self.frequency_of_frequencies.get(1, 0) / total_count if total_count > 0 else 0)
        
        return SmoothingStatistics(
            method="good_turing",
            vocabulary_size=self.vocabulary_size,
            total_ngrams=total_count,
            unseen_probability=unseen_prob,
            smoothing_factor=0.0,  # Not applicable for Good-Turing
            coverage=seen_mass
        )


class SimpleBackoffSmoothing(BaseSmoothingTechnique):
    """
    Simple Backoff Smoothing
    
    Falls back to lower-order n-grams when higher-order n-grams are not found.
    Useful for handling sparse data in character models.
    """
    
    def __init__(self, vocabulary_size: int, backoff_factor: float = 0.4):
        super().__init__(vocabulary_size)
        self.backoff_factor = backoff_factor
        self.logger.info(f"Initialized Simple Backoff smoothing with factor={backoff_factor}")
    
    def smooth_probability(self, ngram_count: int, total_count: int, context_count: int = None) -> float:
        """Calculate backoff-smoothed probability"""
        if ngram_count > 0 and total_count > 0:
            # Use maximum likelihood estimate for seen n-grams
            return ngram_count / total_count
        else:
            # Backoff to uniform distribution with discounting
            return self.backoff_factor / self.vocabulary_size
    
    def unseen_probability(self, total_count: int, context_count: int = None) -> float:
        """Calculate probability for unseen n-grams"""
        return self.backoff_factor / self.vocabulary_size
    
    def get_statistics(self, total_count: int) -> SmoothingStatistics:
        """Get smoothing statistics"""
        unseen_prob = self.unseen_probability(total_count)
        coverage = 1.0 - self.backoff_factor if total_count > 0 else 0.0
        
        return SmoothingStatistics(
            method="simple_backoff",
            vocabulary_size=self.vocabulary_size,
            total_ngrams=total_count,
            unseen_probability=unseen_prob,
            smoothing_factor=self.backoff_factor,
            coverage=coverage
        )


def create_smoothing_technique(
    method: SmoothingMethod, 
    vocabulary_size: int, 
    **kwargs
) -> BaseSmoothingTechnique:
    """
    Factory function to create smoothing techniques
    
    Args:
        method: Smoothing method to use
        vocabulary_size: Size of the character vocabulary
        **kwargs: Method-specific parameters
        
    Returns:
        Configured smoothing technique
    """
    if method == SmoothingMethod.LAPLACE:
        alpha = kwargs.get('alpha', 1.0)
        return LaplaceSmoothing(vocabulary_size, alpha)
    
    elif method == SmoothingMethod.GOOD_TURING:
        use_simple_gt = kwargs.get('use_simple_gt', True)
        return GoodTuringSmoothing(vocabulary_size, use_simple_gt)
    
    elif method == SmoothingMethod.SIMPLE_BACKOFF:
        backoff_factor = kwargs.get('backoff_factor', 0.4)
        return SimpleBackoffSmoothing(vocabulary_size, backoff_factor)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def compare_smoothing_methods(
    ngram_counts: Counter,
    vocabulary_size: int,
    test_ngrams: List[str] = None
) -> Dict[str, SmoothingStatistics]:
    """
    Compare different smoothing methods on the same data
    
    Args:
        ngram_counts: N-gram counts from training data
        vocabulary_size: Size of character vocabulary
        test_ngrams: Optional test n-grams for evaluation
        
    Returns:
        Dictionary of method names to statistics
    """
    methods = [
        SmoothingMethod.LAPLACE,
        SmoothingMethod.GOOD_TURING,
        SmoothingMethod.SIMPLE_BACKOFF
    ]
    
    total_count = sum(ngram_counts.values())
    results = {}
    
    for method in methods:
        smoothing = create_smoothing_technique(method, vocabulary_size)
        
        # Train Good-Turing if needed
        if isinstance(smoothing, GoodTuringSmoothing):
            smoothing.train(ngram_counts)
        
        # Get statistics
        stats = smoothing.get_statistics(total_count)
        results[method.value] = stats
    
    return results


if __name__ == "__main__":
    # Demo usage
    print("🧮 SMOOTHING TECHNIQUES DEMO")
    print("=" * 40)
    
    # Sample n-gram counts (simulated)
    sample_counts = Counter({
        'abc': 100, 'bcd': 80, 'cde': 60, 'def': 40,
        'efg': 20, 'fgh': 10, 'ghi': 5, 'hij': 2, 'ijk': 1
    })
    
    vocabulary_size = 50  # Simulated character vocabulary
    
    # Compare smoothing methods
    comparison = compare_smoothing_methods(sample_counts, vocabulary_size)
    
    print("Method comparison results:")
    for method_name, stats in comparison.items():
        print(f"\n{method_name.upper()}:")
        print(f"  Unseen probability: {stats.unseen_probability:.6f}")
        print(f"  Coverage: {stats.coverage:.3f}")
        print(f"  Smoothing factor: {stats.smoothing_factor:.3f}") 