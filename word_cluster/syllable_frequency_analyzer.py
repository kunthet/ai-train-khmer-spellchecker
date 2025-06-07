"""
Syllable Frequency Analyzer

This module provides comprehensive syllable frequency analysis and modeling
for Khmer text, including frequency distributions, validation systems,
and out-of-vocabulary detection.
"""

import logging
import json
import pickle
from typing import List, Dict, Set, Tuple, Optional, Counter
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import math
import time

from .syllable_api import SyllableSegmentationAPI, SegmentationMethod


@dataclass
class SyllableFrequencyStats:
    """Statistics for syllable frequency analysis"""
    total_syllables: int
    unique_syllables: int
    corpus_size_texts: int
    avg_syllables_per_text: float
    most_common_syllables: List[Tuple[str, int]]
    rare_syllables: List[Tuple[str, int]]
    frequency_distribution: Dict[str, int]
    coverage_percentiles: Dict[int, int]  # percentile -> num syllables
    entropy: float


@dataclass
class SyllableValidationResult:
    """Result of syllable validation"""
    syllable: str
    is_valid: bool
    frequency: int
    confidence_score: float
    rarity_level: str  # common, uncommon, rare, unknown
    suggestions: List[Tuple[str, float]]  # syllable, similarity_score


class SyllableFrequencyAnalyzer:
    """Advanced syllable frequency analysis and validation"""
    
    def __init__(
        self,
        segmentation_method: SegmentationMethod = SegmentationMethod.REGEX_ADVANCED,
        min_frequency_threshold: int = 2,
        rare_frequency_threshold: int = 10,
        common_frequency_threshold: int = 100,
        filter_non_khmer: bool = True,
        keep_punctuation: bool = True,
        min_khmer_ratio: float = 0.5,
        filter_multidigit_numbers: bool = True,
        max_digit_length: int = 1
    ):
        self.segmentation_api = SyllableSegmentationAPI(segmentation_method)
        self.min_frequency_threshold = min_frequency_threshold
        self.rare_frequency_threshold = rare_frequency_threshold
        self.common_frequency_threshold = common_frequency_threshold
        self.filter_non_khmer = filter_non_khmer
        self.keep_punctuation = keep_punctuation
        self.min_khmer_ratio = min_khmer_ratio
        self.filter_multidigit_numbers = filter_multidigit_numbers
        self.max_digit_length = max_digit_length
        
        self.logger = logging.getLogger("syllable_frequency_analyzer")
        
        # Frequency data
        self.syllable_frequencies = Counter()
        self.syllable_contexts = defaultdict(lambda: defaultdict(int))  # syllable -> {prev_syllable: count}
        self.syllable_bigrams = Counter()
        self.syllable_trigrams = Counter()
        
        # Categorized frequency data
        self.khmer_syllable_frequencies = Counter()
        self.non_khmer_syllable_frequencies = Counter()
        self.punctuation_frequencies = Counter()
        self.mixed_syllable_frequencies = Counter()
        self.number_frequencies = Counter()  # New: track numbers separately
        
        # Validation sets
        self.valid_syllables = set()
        self.common_syllables = set()
        self.rare_syllables = set()
        self.unknown_syllables = set()
        
        # Statistics
        self.corpus_stats = None
        self.is_trained = False
        
        self.logger.info(f"SyllableFrequencyAnalyzer initialized with {segmentation_method.value}")
        self.logger.info(f"Filter non-Khmer: {filter_non_khmer}, Keep punctuation: {keep_punctuation}")
        self.logger.info(f"Filter multi-digit numbers: {filter_multidigit_numbers}, Max digit length: {max_digit_length}")
    
    def _is_khmer_char(self, char: str) -> bool:
        """Check if character is Khmer (including numbers and symbols)"""
        return '\u1780' <= char <= '\u17FF'
    
    def _is_khmer_digit(self, char: str) -> bool:
        """Check if character is Khmer digit (០-៩)"""
        return '\u17E0' <= char <= '\u17E9'
    
    def _is_arabic_digit(self, char: str) -> bool:
        """Check if character is Arabic digit (0-9)"""
        return '0' <= char <= '9'
    
    def _is_any_digit(self, char: str) -> bool:
        """Check if character is any digit (Khmer or Arabic)"""
        return self._is_khmer_digit(char) or self._is_arabic_digit(char)
    
    def _is_punctuation(self, syllable: str) -> bool:
        """Check if syllable is punctuation or whitespace"""
        import string
        return all(c in string.punctuation + string.whitespace + '។៕៖ៗ៘៙៚៛ៜ៝៞៟' for c in syllable)
    
    def _is_number_sequence(self, syllable: str) -> Tuple[bool, str]:
        """
        Check if syllable is a number sequence and classify it
        
        Returns:
            (is_number, number_type) where number_type is:
            'single_khmer', 'single_arabic', 'multi_khmer', 'multi_arabic', 'mixed_digits', 'not_number'
        """
        syllable = syllable.strip()
        if not syllable:
            return False, 'not_number'
        
        # Count different types of digits
        khmer_digits = sum(1 for c in syllable if self._is_khmer_digit(c))
        arabic_digits = sum(1 for c in syllable if self._is_arabic_digit(c))
        other_chars = len(syllable) - khmer_digits - arabic_digits
        
        # Must be all digits to be considered a number sequence
        if other_chars > 0:
            return False, 'not_number'
        
        total_digits = khmer_digits + arabic_digits
        if total_digits == 0:
            return False, 'not_number'
        
        # Classify number type
        if total_digits == 1:
            if khmer_digits == 1:
                return True, 'single_khmer'
            else:
                return True, 'single_arabic'
        else:  # Multi-digit
            if khmer_digits > 0 and arabic_digits > 0:
                return True, 'mixed_digits'
            elif khmer_digits > 0:
                return True, 'multi_khmer'
            else:
                return True, 'multi_arabic'
    
    def _classify_syllable(self, syllable: str) -> str:
        """
        Classify syllable type
        
        Returns:
            'khmer': Pure Khmer syllable
            'mixed': Contains both Khmer and non-Khmer characters
            'non_khmer': Pure non-Khmer (English, numbers, etc.)
            'punctuation': Punctuation or whitespace
            'number': Number sequence (handled separately)
            'empty': Empty or whitespace only
        """
        syllable = syllable.strip()
        
        if not syllable:
            return 'empty'
        
        if self._is_punctuation(syllable):
            return 'punctuation'
        
        # Check if it's a number sequence
        is_number, number_type = self._is_number_sequence(syllable)
        if is_number:
            return 'number'
        
        khmer_count = sum(1 for c in syllable if self._is_khmer_char(c))
        total_count = len(syllable)
        khmer_ratio = khmer_count / total_count if total_count > 0 else 0
        
        if khmer_ratio == 1.0:
            return 'khmer'
        elif khmer_ratio >= self.min_khmer_ratio:
            return 'mixed'
        elif khmer_ratio > 0:
            return 'mixed'
        else:
            return 'non_khmer'
    
    def _should_include_syllable(self, syllable: str, syllable_type: str) -> bool:
        """Determine if syllable should be included in frequency analysis"""
        if not self.filter_non_khmer and not self.filter_multidigit_numbers:
            return True  # Include everything if filtering is disabled
        
        if syllable_type == 'empty':
            return False
        
        if syllable_type == 'khmer':
            return True
        
        if syllable_type == 'mixed':
            return True  # Keep mixed content as it's often legitimate
        
        if syllable_type == 'punctuation':
            return self.keep_punctuation
        
        if syllable_type == 'number':
            return self._should_include_number(syllable)
        
        if syllable_type == 'non_khmer':
            return False if self.filter_non_khmer else True
        
        return True
    
    def _should_include_number(self, syllable: str) -> bool:
        """Determine if a number sequence should be included"""
        if not self.filter_multidigit_numbers:
            return True
        
        is_number, number_type = self._is_number_sequence(syllable)
        if not is_number:
            return True
        
        # Allow single digits of any type
        if number_type in ['single_khmer', 'single_arabic']:
            return len(syllable) <= self.max_digit_length
        
        # Filter multi-digit numbers
        if number_type in ['multi_khmer', 'multi_arabic', 'mixed_digits']:
            return len(syllable) <= self.max_digit_length
        
        return True
    
    def train_on_texts(
        self, 
        texts: List[str], 
        show_progress: bool = True,
        save_to_file: Optional[str] = None
    ) -> SyllableFrequencyStats:
        """
        Train the frequency analyzer on a corpus of texts
        
        Args:
            texts: List of texts to analyze
            show_progress: Whether to show progress updates
            save_to_file: Optional file path to save frequency data
            
        Returns:
            SyllableFrequencyStats with analysis results
        """
        self.logger.info(f"Training syllable frequency analyzer on {len(texts)} texts")
        start_time = time.time()
        
        # Segment all texts
        batch_result = self.segmentation_api.segment_batch(texts, show_progress=show_progress)
        
        successful_results = [r for r in batch_result.results if r.success]
        self.logger.info(f"Successfully segmented {len(successful_results)}/{len(texts)} texts")
        
        # Build frequency tables
        for result in successful_results:
            syllables = result.syllables
            
            # Filter and categorize syllables
            filtered_syllables = []
            for syllable in syllables:
                if syllable.strip():  # Skip empty syllables
                    syllable_type = self._classify_syllable(syllable)
                    
                    # Always track in categorized counters for statistics
                    if syllable_type == 'khmer':
                        self.khmer_syllable_frequencies[syllable] += 1
                    elif syllable_type == 'non_khmer':
                        self.non_khmer_syllable_frequencies[syllable] += 1
                    elif syllable_type == 'punctuation':
                        self.punctuation_frequencies[syllable] += 1
                    elif syllable_type == 'number':
                        self.number_frequencies[syllable] += 1
                    else:
                        self.mixed_syllable_frequencies[syllable] += 1
                    
                    # Only include in main frequency table if passes filter
                    if self._should_include_syllable(syllable, syllable_type):
                        self.syllable_frequencies[syllable] += 1
                        filtered_syllables.append(syllable)
            
            # Update bigrams (only with filtered syllables)
            for i in range(len(filtered_syllables) - 1):
                if filtered_syllables[i].strip() and filtered_syllables[i+1].strip():
                    bigram = (filtered_syllables[i], filtered_syllables[i+1])
                    self.syllable_bigrams[bigram] += 1
            
            # Update trigrams (only with filtered syllables)
            for i in range(len(filtered_syllables) - 2):
                if all(s.strip() for s in filtered_syllables[i:i+3]):
                    trigram = (filtered_syllables[i], filtered_syllables[i+1], filtered_syllables[i+2])
                    self.syllable_trigrams[trigram] += 1
            
            # Update context information (only with filtered syllables)
            for i in range(1, len(filtered_syllables)):
                if filtered_syllables[i].strip() and filtered_syllables[i-1].strip():
                    self.syllable_contexts[filtered_syllables[i]][filtered_syllables[i-1]] += 1
        
        # Categorize syllables by frequency
        self._categorize_syllables()
        
        # Generate statistics
        self.corpus_stats = self._generate_statistics(len(successful_results))
        self.is_trained = True
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f}s")
        
        # Save to file if requested
        if save_to_file:
            self.save_frequency_data(save_to_file)
        
        return self.corpus_stats
    
    def _categorize_syllables(self):
        """Categorize syllables by frequency levels"""
        self.valid_syllables = set()
        self.common_syllables = set()
        self.rare_syllables = set()
        
        for syllable, frequency in self.syllable_frequencies.items():
            if frequency >= self.min_frequency_threshold:
                self.valid_syllables.add(syllable)
                
                if frequency >= self.common_frequency_threshold:
                    self.common_syllables.add(syllable)
                elif frequency <= self.rare_frequency_threshold:
                    self.rare_syllables.add(syllable)
    
    def _generate_statistics(self, corpus_size: int) -> SyllableFrequencyStats:
        """Generate comprehensive frequency statistics"""
        total_syllables = sum(self.syllable_frequencies.values())
        unique_syllables = len(self.syllable_frequencies)
        
        # Most and least common syllables
        most_common = self.syllable_frequencies.most_common(50)
        rare_syllables = [(syll, freq) for syll, freq in self.syllable_frequencies.items() 
                         if freq <= self.rare_frequency_threshold][:50]
        
        # Coverage percentiles (how many syllables cover X% of usage)
        coverage_percentiles = self._calculate_coverage_percentiles()
        
        # Calculate entropy
        entropy = self._calculate_entropy()
        
        return SyllableFrequencyStats(
            total_syllables=total_syllables,
            unique_syllables=unique_syllables,
            corpus_size_texts=corpus_size,
            avg_syllables_per_text=total_syllables / corpus_size if corpus_size > 0 else 0,
            most_common_syllables=most_common,
            rare_syllables=rare_syllables,
            frequency_distribution=dict(self.syllable_frequencies),
            coverage_percentiles=coverage_percentiles,
            entropy=entropy
        )
    
    def _calculate_coverage_percentiles(self) -> Dict[int, int]:
        """Calculate how many syllables cover each percentage of usage"""
        total_count = sum(self.syllable_frequencies.values())
        sorted_frequencies = self.syllable_frequencies.most_common()
        
        coverage_percentiles = {}
        cumulative_count = 0
        
        for i, (syllable, frequency) in enumerate(sorted_frequencies):
            cumulative_count += frequency
            coverage_percentage = (cumulative_count / total_count) * 100
            
            # Record percentile milestones
            for percentile in [50, 75, 90, 95, 99]:
                if percentile not in coverage_percentiles and coverage_percentage >= percentile:
                    coverage_percentiles[percentile] = i + 1
        
        return coverage_percentiles
    
    def _calculate_entropy(self) -> float:
        """Calculate Shannon entropy of syllable distribution"""
        if not self.syllable_frequencies:
            return 0.0
        
        total = sum(self.syllable_frequencies.values())
        entropy = 0.0
        
        for frequency in self.syllable_frequencies.values():
            if frequency > 0:
                prob = frequency / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def validate_syllable(self, syllable: str) -> SyllableValidationResult:
        """
        Validate a single syllable and provide detailed analysis
        
        Args:
            syllable: Syllable to validate
            
        Returns:
            SyllableValidationResult with validation details
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before validation")
        
        frequency = self.syllable_frequencies.get(syllable, 0)
        is_valid = syllable in self.valid_syllables
        
        # Determine rarity level
        if syllable in self.common_syllables:
            rarity_level = "common"
        elif syllable in self.rare_syllables:
            rarity_level = "rare"
        elif syllable in self.valid_syllables:
            rarity_level = "uncommon"
        else:
            rarity_level = "unknown"
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(syllable, frequency)
        
        # Generate suggestions for unknown syllables
        suggestions = []
        if not is_valid:
            suggestions = self._generate_suggestions(syllable)
        
        return SyllableValidationResult(
            syllable=syllable,
            is_valid=is_valid,
            frequency=frequency,
            confidence_score=confidence_score,
            rarity_level=rarity_level,
            suggestions=suggestions
        )
    
    def _calculate_confidence_score(self, syllable: str, frequency: int) -> float:
        """Calculate confidence score for a syllable (0-1)"""
        if frequency == 0:
            return 0.0
        
        # Normalize frequency to 0-1 scale
        max_frequency = max(self.syllable_frequencies.values()) if self.syllable_frequencies else 1
        frequency_score = min(frequency / max_frequency, 1.0)
        
        # Consider context information
        context_score = 0.0
        if syllable in self.syllable_contexts:
            context_count = sum(self.syllable_contexts[syllable].values())
            context_score = min(context_count / 100, 1.0)  # Normalize to 100 contexts
        
        # Weighted combination
        confidence = 0.7 * frequency_score + 0.3 * context_score
        return min(confidence, 1.0)
    
    def _generate_suggestions(self, syllable: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Generate syllable suggestions based on similarity"""
        suggestions = []
        
        # Simple character-based similarity
        for known_syllable in self.valid_syllables:
            similarity = self._calculate_syllable_similarity(syllable, known_syllable)
            if similarity > 0.5:  # Only suggest reasonably similar syllables
                suggestions.append((known_syllable, similarity))
        
        # Sort by similarity and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]
    
    def _calculate_syllable_similarity(self, syllable1: str, syllable2: str) -> float:
        """Calculate similarity between two syllables (0-1)"""
        if syllable1 == syllable2:
            return 1.0
        
        # Simple Levenshtein-based similarity
        max_len = max(len(syllable1), len(syllable2))
        if max_len == 0:
            return 1.0
        
        # Count matching characters at same positions
        matches = sum(1 for i in range(min(len(syllable1), len(syllable2))) 
                     if syllable1[i] == syllable2[i])
        
        return matches / max_len
    
    def validate_text(self, text: str) -> Dict[str, any]:
        """
        Validate entire text and return detailed analysis
        
        Args:
            text: Text to validate
            
        Returns:
            Dictionary with validation results
        """
        result = self.segmentation_api.segment_text(text)
        
        if not result.success:
            return {
                "success": False,
                "error": result.error_message,
                "syllables": []
            }
        
        syllable_validations = []
        total_syllables = 0
        valid_syllables = 0
        unknown_syllables = []
        
        for syllable in result.syllables:
            if syllable.strip():
                validation = self.validate_syllable(syllable)
                syllable_validations.append(validation)
                total_syllables += 1
                
                if validation.is_valid:
                    valid_syllables += 1
                else:
                    unknown_syllables.append(syllable)
        
        return {
            "success": True,
            "text": text,
            "total_syllables": total_syllables,
            "valid_syllables": valid_syllables,
            "unknown_syllables": unknown_syllables,
            "validity_rate": valid_syllables / total_syllables if total_syllables > 0 else 0,
            "syllable_validations": syllable_validations,
            "segmentation_time": result.processing_time
        }
    
    def get_frequency_stats(self) -> Optional[SyllableFrequencyStats]:
        """Get current frequency statistics"""
        return self.corpus_stats
    
    def save_frequency_data(self, filepath: str):
        """Save frequency data to file"""
        data = {
            "syllable_frequencies": dict(self.syllable_frequencies),
            "syllable_bigrams": {f"{b[0]}|{b[1]}": count for b, count in self.syllable_bigrams.items()},
            "syllable_trigrams": {f"{t[0]}|{t[1]}|{t[2]}": count for t, count in self.syllable_trigrams.items()},
            "syllable_contexts": {syll: dict(contexts) for syll, contexts in self.syllable_contexts.items()},
            
            # Categorized frequencies
            "khmer_syllable_frequencies": dict(self.khmer_syllable_frequencies),
            "non_khmer_syllable_frequencies": dict(self.non_khmer_syllable_frequencies),
            "punctuation_frequencies": dict(self.punctuation_frequencies),
            "mixed_syllable_frequencies": dict(self.mixed_syllable_frequencies),
            "number_frequencies": dict(self.number_frequencies),
            
            # Validation sets
            "valid_syllables": list(self.valid_syllables),
            "common_syllables": list(self.common_syllables),
            "rare_syllables": list(self.rare_syllables),
            
            # Configuration
            "thresholds": {
                "min_frequency": self.min_frequency_threshold,
                "rare_frequency": self.rare_frequency_threshold,
                "common_frequency": self.common_frequency_threshold
            },
            "filtering_config": {
                "filter_non_khmer": self.filter_non_khmer,
                "keep_punctuation": self.keep_punctuation,
                "min_khmer_ratio": self.min_khmer_ratio,
                "filter_multidigit_numbers": self.filter_multidigit_numbers,
                "max_digit_length": self.max_digit_length
            },
            "corpus_stats": self.corpus_stats.__dict__ if self.corpus_stats else None
        }
        
        # Save as JSON
        json_path = f"{filepath}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Save as pickle for faster loading
        pickle_path = f"{filepath}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Frequency data saved to {json_path} and {pickle_path}")
    
    def load_frequency_data(self, filepath: str):
        """Load frequency data from file"""
        # Try pickle first (faster)
        pickle_path = f"{filepath}.pkl"
        if Path(pickle_path).exists():
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
        else:
            # Fall back to JSON
            json_path = f"{filepath}.json"
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # Restore frequency data
        self.syllable_frequencies = Counter(data["syllable_frequencies"])
        
        # Restore categorized frequencies (with backward compatibility)
        self.khmer_syllable_frequencies = Counter(data.get("khmer_syllable_frequencies", {}))
        self.non_khmer_syllable_frequencies = Counter(data.get("non_khmer_syllable_frequencies", {}))
        self.punctuation_frequencies = Counter(data.get("punctuation_frequencies", {}))
        self.mixed_syllable_frequencies = Counter(data.get("mixed_syllable_frequencies", {}))
        self.number_frequencies = Counter(data.get("number_frequencies", {}))
        
        # Restore bigrams
        self.syllable_bigrams = Counter()
        for bigram_str, count in data["syllable_bigrams"].items():
            parts = bigram_str.split("|")
            self.syllable_bigrams[(parts[0], parts[1])] = count
        
        # Restore trigrams
        self.syllable_trigrams = Counter()
        for trigram_str, count in data["syllable_trigrams"].items():
            parts = trigram_str.split("|")
            self.syllable_trigrams[(parts[0], parts[1], parts[2])] = count
        
        # Restore contexts
        self.syllable_contexts = defaultdict(lambda: defaultdict(int))
        for syllable, contexts in data["syllable_contexts"].items():
            for prev_syll, count in contexts.items():
                self.syllable_contexts[syllable][prev_syll] = count
        
        # Restore syllable sets
        self.valid_syllables = set(data["valid_syllables"])
        self.common_syllables = set(data["common_syllables"])
        self.rare_syllables = set(data["rare_syllables"])
        
        # Restore thresholds
        thresholds = data["thresholds"]
        self.min_frequency_threshold = thresholds["min_frequency"]
        self.rare_frequency_threshold = thresholds["rare_frequency"]
        self.common_frequency_threshold = thresholds["common_frequency"]
        
        # Restore filtering configuration (with backward compatibility)
        filtering_config = data.get("filtering_config", {})
        self.filter_non_khmer = filtering_config.get("filter_non_khmer", True)
        self.keep_punctuation = filtering_config.get("keep_punctuation", True)
        self.min_khmer_ratio = filtering_config.get("min_khmer_ratio", 0.5)
        self.filter_multidigit_numbers = filtering_config.get("filter_multidigit_numbers", True)
        self.max_digit_length = filtering_config.get("max_digit_length", 1)
        
        # Restore statistics
        if data["corpus_stats"]:
            stats_data = data["corpus_stats"]
            self.corpus_stats = SyllableFrequencyStats(**stats_data)
        
        self.is_trained = True
        self.logger.info(f"Frequency data loaded from {filepath}")
        self.logger.info(f"Filter non-Khmer: {self.filter_non_khmer}, Keep punctuation: {self.keep_punctuation}")
    
    def generate_report(self) -> str:
        """Generate human-readable frequency analysis report"""
        if not self.is_trained:
            return "❌ Analyzer not trained yet"
        
        stats = self.corpus_stats
        
        report = []
        report.append("📊 SYLLABLE FREQUENCY ANALYSIS REPORT")
        report.append("=" * 45)
        report.append("")
        
        # Overview
        report.append("📋 OVERVIEW")
        report.append("-" * 15)
        report.append(f"Total syllables processed: {stats.total_syllables:,}")
        report.append(f"Unique syllables found: {stats.unique_syllables:,}")
        report.append(f"Corpus size (texts): {stats.corpus_size_texts:,}")
        report.append(f"Average syllables per text: {stats.avg_syllables_per_text:.1f}")
        report.append(f"Syllable entropy: {stats.entropy:.3f}")
        
        # Filtering statistics
        if self.filter_non_khmer:
            total_before_filtering = (sum(self.khmer_syllable_frequencies.values()) + 
                                    sum(self.non_khmer_syllable_frequencies.values()) + 
                                    sum(self.punctuation_frequencies.values()) + 
                                    sum(self.mixed_syllable_frequencies.values()) +
                                    sum(self.number_frequencies.values()))
            
            report.append("")
            report.append("🔍 FILTERING STATISTICS")
            report.append("-" * 23)
            report.append(f"Total syllables before filtering: {total_before_filtering:,}")
            report.append(f"Khmer syllables: {sum(self.khmer_syllable_frequencies.values()):,}")
            report.append(f"Mixed syllables: {sum(self.mixed_syllable_frequencies.values()):,}")
            report.append(f"Non-Khmer syllables (filtered): {sum(self.non_khmer_syllable_frequencies.values()):,}")
            report.append(f"Punctuation ({'kept' if self.keep_punctuation else 'filtered'}): {sum(self.punctuation_frequencies.values()):,}")
            report.append(f"Number sequences ({'kept' if self.filter_multidigit_numbers else 'filtered'}): {sum(self.number_frequencies.values()):,}")
            filter_ratio = stats.total_syllables / total_before_filtering if total_before_filtering > 0 else 0
            report.append(f"Filtering ratio: {filter_ratio:.1%} kept")
        
        report.append("")
        
        # Frequency categories
        report.append("🏷️  FREQUENCY CATEGORIES")
        report.append("-" * 25)
        report.append(f"Valid syllables (≥{self.min_frequency_threshold}): {len(self.valid_syllables):,}")
        report.append(f"Common syllables (≥{self.common_frequency_threshold}): {len(self.common_syllables):,}")
        report.append(f"Rare syllables (≤{self.rare_frequency_threshold}): {len(self.rare_syllables):,}")
        report.append("")
        
        # Coverage statistics
        report.append("📈 COVERAGE STATISTICS")
        report.append("-" * 22)
        for percentile, syllable_count in sorted(stats.coverage_percentiles.items()):
            report.append(f"{percentile}% of usage covered by top {syllable_count:,} syllables")
        report.append("")
        
        # Most common syllables
        report.append("🔝 TOP 10 MOST COMMON SYLLABLES")
        report.append("-" * 32)
        for i, (syllable, frequency) in enumerate(stats.most_common_syllables[:10], 1):
            percentage = (frequency / stats.total_syllables) * 100
            syllable_type = self._classify_syllable(syllable)
            type_indicator = {"khmer": "🇰🇭", "mixed": "🔄", "punctuation": "��", "non_khmer": "🌐", "number": "🔢"}.get(syllable_type, "❓")
            report.append(f"{i:2d}. '{syllable}' {type_indicator}: {frequency:,} ({percentage:.2f}%)")
        report.append("")
        
        # Syllable type breakdown
        if self.filter_non_khmer or self.filter_multidigit_numbers:
            report.append("📊 SYLLABLE TYPE BREAKDOWN (Top 5 each)")
            report.append("-" * 35)
            
            if self.khmer_syllable_frequencies:
                report.append("🇰🇭 Khmer syllables:")
                for i, (syll, freq) in enumerate(self.khmer_syllable_frequencies.most_common(5), 1):
                    report.append(f"   {i}. '{syll}': {freq:,}")
            
            if self.mixed_syllable_frequencies:
                report.append("🔄 Mixed syllables:")
                for i, (syll, freq) in enumerate(self.mixed_syllable_frequencies.most_common(5), 1):
                    report.append(f"   {i}. '{syll}': {freq:,}")
            
            if self.number_frequencies:
                numbers_status = "filtered out" if self.filter_multidigit_numbers else "included"
                report.append(f"🔢 Number sequences ({numbers_status}):")
                for i, (syll, freq) in enumerate(self.number_frequencies.most_common(5), 1):
                    is_number, number_type = self._is_number_sequence(syll)
                    type_desc = {
                        'single_khmer': 'single Khmer',
                        'single_arabic': 'single Arabic',
                        'multi_khmer': 'multi Khmer',
                        'multi_arabic': 'multi Arabic',
                        'mixed_digits': 'mixed digits'
                    }.get(number_type, 'unknown')
                    report.append(f"   {i}. '{syll}' ({type_desc}): {freq:,}")
            
            if self.non_khmer_syllable_frequencies:
                report.append("🌐 Non-Khmer syllables (filtered out):")
                for i, (syll, freq) in enumerate(self.non_khmer_syllable_frequencies.most_common(5), 1):
                    report.append(f"   {i}. '{syll}': {freq:,}")
            
            report.append("")
        
        # Bigram analysis
        if self.syllable_bigrams:
            report.append("🔗 TOP 5 SYLLABLE BIGRAMS")
            report.append("-" * 25)
            top_bigrams = self.syllable_bigrams.most_common(5)
            for i, (bigram, frequency) in enumerate(top_bigrams, 1):
                report.append(f"{i}. '{bigram[0]}' → '{bigram[1]}': {frequency:,}")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Demo usage
    analyzer = SyllableFrequencyAnalyzer()
    
    # Example texts for testing
    sample_texts = [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
        "ខ្ញុំស្រលាញ់ភាសាខ្មែរណាស់។",
        "ប្រទេសកម្ពុជាមានទីរដ្ឋធានីភ្នំពេញ។"
    ]
    
    print("🔍 Training syllable frequency analyzer...")
    stats = analyzer.train_on_texts(sample_texts, show_progress=False)
    
    print(f"\n📊 Analysis Results:")
    print(f"Total syllables: {stats.total_syllables}")
    print(f"Unique syllables: {stats.unique_syllables}")
    print(f"Most common: {stats.most_common_syllables[:5]}")
    
    # Test validation
    print(f"\n🧪 Testing validation...")
    test_syllable = "នេះ"
    validation = analyzer.validate_syllable(test_syllable)
    print(f"Syllable '{test_syllable}': Valid={validation.is_valid}, Frequency={validation.frequency}")
    
    # Generate report
    print(f"\n📋 Full Report:")
    print(analyzer.generate_report()) 