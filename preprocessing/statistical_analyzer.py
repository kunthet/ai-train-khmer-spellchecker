"""
Statistical Analysis Module

This module provides comprehensive statistical analysis tools for Khmer text,
including character, syllable, and word frequency analysis specifically
designed for spellchecker development.
"""

import logging
from typing import List, Dict, Counter, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path
import math


@dataclass
class CharacterStatistics:
    """Statistics for character-level analysis"""
    total_characters: int
    unique_characters: int
    khmer_characters: int
    non_khmer_characters: int
    character_frequencies: Dict[str, int]
    bigram_frequencies: Dict[str, int]
    trigram_frequencies: Dict[str, int]
    khmer_ratio: float
    most_common_chars: List[Tuple[str, int]]
    rare_characters: List[Tuple[str, int]]


@dataclass
class SyllableStatistics:
    """Statistics for syllable-level analysis"""
    total_syllables: int
    unique_syllables: int
    syllable_frequencies: Dict[str, int]
    syllable_bigrams: Dict[str, int]
    avg_syllable_length: float
    syllable_length_distribution: Dict[int, int]
    most_common_syllables: List[Tuple[str, int]]
    rare_syllables: List[Tuple[str, int]]
    compound_syllables: Dict[str, int]


@dataclass
class WordStatistics:
    """Statistics for word-level analysis"""
    total_words: int
    unique_words: int
    word_frequencies: Dict[str, int]
    word_bigrams: Dict[str, int]
    avg_word_length: float
    word_length_distribution: Dict[int, int]
    most_common_words: List[Tuple[str, int]]
    rare_words: List[Tuple[str, int]]
    compound_words: Dict[str, int]


@dataclass
class CorpusStatistics:
    """Overall corpus statistics"""
    total_texts: int
    total_size_bytes: int
    character_stats: CharacterStatistics
    syllable_stats: SyllableStatistics
    word_stats: WordStatistics
    vocabulary_coverage: float
    text_diversity: float
    quality_score: float


class StatisticalAnalyzer:
    """Advanced statistical analysis for Khmer text corpus"""
    
    def __init__(
        self,
        min_frequency: int = 2,
        max_ngram_size: int = 3,
        rare_threshold: int = 5,
        common_threshold: int = 100
    ):
        self.min_frequency = min_frequency
        self.max_ngram_size = max_ngram_size
        self.rare_threshold = rare_threshold
        self.common_threshold = common_threshold
        
        self.logger = logging.getLogger("statistical_analyzer")
        
        # Initialize counters
        self.character_counter = Counter()
        self.syllable_counter = Counter()
        self.word_counter = Counter()
        
        # N-gram counters
        self.char_bigrams = Counter()
        self.char_trigrams = Counter()
        self.syllable_bigrams = Counter()
        self.word_bigrams = Counter()
        
        # Additional statistics
        self.syllable_lengths = Counter()
        self.word_lengths = Counter()
        
        self.logger.info("StatisticalAnalyzer initialized")
    
    def _is_khmer_char(self, char: str) -> bool:
        """Check if character is Khmer"""
        return '\u1780' <= char <= '\u17FF'
    
    def _generate_ngrams(self, sequence: List[str], n: int) -> List[str]:
        """Generate n-grams from a sequence"""
        if len(sequence) < n:
            return []
        return [' '.join(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    
    def analyze_text(self, text: str, syllables: List[str] = None):
        """Analyze a single text and update statistics"""
        if not text.strip():
            return
        
        # Character-level analysis
        for char in text:
            if char.strip():  # Skip whitespace
                self.character_counter[char] += 1
        
        # Character n-grams
        char_list = [c for c in text if c.strip()]
        if len(char_list) >= 2:
            bigrams = self._generate_ngrams(char_list, 2)
            for bigram in bigrams:
                self.char_bigrams[bigram] += 1
        
        if len(char_list) >= 3:
            trigrams = self._generate_ngrams(char_list, 3)
            for trigram in trigrams:
                self.char_trigrams[trigram] += 1
        
        # Syllable-level analysis (if provided)
        if syllables:
            for syllable in syllables:
                if syllable.strip():
                    self.syllable_counter[syllable] += 1
                    self.syllable_lengths[len(syllable)] += 1
            
            # Syllable bigrams
            if len(syllables) >= 2:
                syll_bigrams = self._generate_ngrams(syllables, 2)
                for bigram in syll_bigrams:
                    self.syllable_bigrams[bigram] += 1
        
        # Word-level analysis
        words = text.split()
        for word in words:
            word = word.strip()
            if word:
                self.word_counter[word] += 1
                self.word_lengths[len(word)] += 1
        
        # Word bigrams
        if len(words) >= 2:
            word_bigrams = self._generate_ngrams(words, 2)
            for bigram in word_bigrams:
                self.word_bigrams[bigram] += 1
    
    def analyze_corpus(self, texts: List[str], syllable_sequences: List[List[str]] = None):
        """Analyze entire corpus"""
        self.logger.info(f"Analyzing corpus of {len(texts)} texts")
        
        if syllable_sequences is None:
            syllable_sequences = [None] * len(texts)
        
        for i, (text, syllables) in enumerate(zip(texts, syllable_sequences)):
            self.analyze_text(text, syllables)
            
            if (i + 1) % 1000 == 0:
                self.logger.info(f"Analyzed {i + 1}/{len(texts)} texts")
        
        self.logger.info("Corpus analysis completed")
    
    def get_character_statistics(self) -> CharacterStatistics:
        """Generate character-level statistics"""
        total_chars = sum(self.character_counter.values())
        unique_chars = len(self.character_counter)
        
        # Count Khmer vs non-Khmer characters
        khmer_count = sum(count for char, count in self.character_counter.items() 
                         if self._is_khmer_char(char))
        non_khmer_count = total_chars - khmer_count
        
        khmer_ratio = khmer_count / total_chars if total_chars > 0 else 0
        
        # Most and least common characters
        most_common = self.character_counter.most_common(20)
        rare_chars = [(char, count) for char, count in self.character_counter.items() 
                     if count <= self.rare_threshold]
        
        return CharacterStatistics(
            total_characters=total_chars,
            unique_characters=unique_chars,
            khmer_characters=khmer_count,
            non_khmer_characters=non_khmer_count,
            character_frequencies=dict(self.character_counter),
            bigram_frequencies=dict(self.char_bigrams),
            trigram_frequencies=dict(self.char_trigrams),
            khmer_ratio=khmer_ratio,
            most_common_chars=most_common,
            rare_characters=rare_chars[:20]  # Top 20 rare characters
        )
    
    def get_syllable_statistics(self) -> SyllableStatistics:
        """Generate syllable-level statistics"""
        total_syllables = sum(self.syllable_counter.values())
        unique_syllables = len(self.syllable_counter)
        
        # Average syllable length
        total_length = sum(len(syllable) * count for syllable, count in self.syllable_counter.items())
        avg_length = total_length / total_syllables if total_syllables > 0 else 0
        
        # Most and least common syllables
        most_common = self.syllable_counter.most_common(50)
        rare_syllables = [(syll, count) for syll, count in self.syllable_counter.items() 
                         if count <= self.rare_threshold]
        
        # Identify compound syllables (containing multiple Khmer characters)
        compound_syllables = {syll: count for syll, count in self.syllable_counter.items() 
                            if len([c for c in syll if self._is_khmer_char(c)]) > 2}
        
        return SyllableStatistics(
            total_syllables=total_syllables,
            unique_syllables=unique_syllables,
            syllable_frequencies=dict(self.syllable_counter),
            syllable_bigrams=dict(self.syllable_bigrams),
            avg_syllable_length=avg_length,
            syllable_length_distribution=dict(self.syllable_lengths),
            most_common_syllables=most_common,
            rare_syllables=rare_syllables[:50],
            compound_syllables=compound_syllables
        )
    
    def get_word_statistics(self) -> WordStatistics:
        """Generate word-level statistics"""
        total_words = sum(self.word_counter.values())
        unique_words = len(self.word_counter)
        
        # Average word length
        total_length = sum(len(word) * count for word, count in self.word_counter.items())
        avg_length = total_length / total_words if total_words > 0 else 0
        
        # Most and least common words
        most_common = self.word_counter.most_common(100)
        rare_words = [(word, count) for word, count in self.word_counter.items() 
                     if count <= self.rare_threshold]
        
        # Identify compound words (containing spaces or multiple syllables)
        compound_words = {word: count for word, count in self.word_counter.items() 
                         if ' ' in word or len(word) > 15}
        
        return WordStatistics(
            total_words=total_words,
            unique_words=unique_words,
            word_frequencies=dict(self.word_counter),
            word_bigrams=dict(self.word_bigrams),
            avg_word_length=avg_length,
            word_length_distribution=dict(self.word_lengths),
            most_common_words=most_common,
            rare_words=rare_words[:100],
            compound_words=compound_words
        )
    
    def get_corpus_statistics(self, total_texts: int, total_size: int) -> CorpusStatistics:
        """Generate overall corpus statistics"""
        char_stats = self.get_character_statistics()
        syllable_stats = self.get_syllable_statistics()
        word_stats = self.get_word_statistics()
        
        # Calculate vocabulary coverage (unique words / total words)
        vocab_coverage = word_stats.unique_words / word_stats.total_words if word_stats.total_words > 0 else 0
        
        # Calculate text diversity (using entropy-based measure)
        text_diversity = self._calculate_diversity(word_stats.word_frequencies)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(char_stats, syllable_stats, word_stats)
        
        return CorpusStatistics(
            total_texts=total_texts,
            total_size_bytes=total_size,
            character_stats=char_stats,
            syllable_stats=syllable_stats,
            word_stats=word_stats,
            vocabulary_coverage=vocab_coverage,
            text_diversity=text_diversity,
            quality_score=quality_score
        )
    
    def _calculate_diversity(self, frequency_dict: Dict[str, int]) -> float:
        """Calculate Shannon entropy for text diversity"""
        if not frequency_dict:
            return 0.0
        
        total = sum(frequency_dict.values())
        entropy = 0.0
        
        for count in frequency_dict.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(frequency_dict)) if frequency_dict else 1
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _calculate_quality_score(
        self, 
        char_stats: CharacterStatistics,
        syllable_stats: SyllableStatistics, 
        word_stats: WordStatistics
    ) -> float:
        """Calculate overall corpus quality score"""
        score = 0.0
        
        # Khmer ratio component (30% weight)
        score += 0.3 * char_stats.khmer_ratio
        
        # Vocabulary richness (25% weight)
        vocab_richness = min(word_stats.unique_words / 10000, 1.0)  # Normalize to 10k words
        score += 0.25 * vocab_richness
        
        # Syllable diversity (25% weight)
        syllable_richness = min(syllable_stats.unique_syllables / 5000, 1.0)  # Normalize to 5k syllables
        score += 0.25 * syllable_richness
        
        # Character diversity (20% weight)
        char_richness = min(char_stats.unique_characters / 200, 1.0)  # Normalize to 200 chars
        score += 0.2 * char_richness
        
        return min(score, 1.0)
    
    def save_statistics(self, output_path: str, corpus_stats: CorpusStatistics):
        """Save statistical analysis to files"""
        output_path = Path(output_path)
        
        # Save complete statistics as pickle
        with open(f"{output_path}_full_stats.pkl", 'wb') as f:
            pickle.dump(corpus_stats, f)
        
        # Save summary as JSON
        summary = {
            'total_texts': corpus_stats.total_texts,
            'total_size_mb': corpus_stats.total_size_bytes / (1024 * 1024),
            'character_stats': {
                'total_characters': corpus_stats.character_stats.total_characters,
                'unique_characters': corpus_stats.character_stats.unique_characters,
                'khmer_ratio': corpus_stats.character_stats.khmer_ratio,
                'most_common_chars': corpus_stats.character_stats.most_common_chars[:10]
            },
            'syllable_stats': {
                'total_syllables': corpus_stats.syllable_stats.total_syllables,
                'unique_syllables': corpus_stats.syllable_stats.unique_syllables,
                'avg_syllable_length': corpus_stats.syllable_stats.avg_syllable_length,
                'most_common_syllables': corpus_stats.syllable_stats.most_common_syllables[:20]
            },
            'word_stats': {
                'total_words': corpus_stats.word_stats.total_words,
                'unique_words': corpus_stats.word_stats.unique_words,
                'avg_word_length': corpus_stats.word_stats.avg_word_length,
                'most_common_words': corpus_stats.word_stats.most_common_words[:20]
            },
            'quality_metrics': {
                'vocabulary_coverage': corpus_stats.vocabulary_coverage,
                'text_diversity': corpus_stats.text_diversity,
                'quality_score': corpus_stats.quality_score
            }
        }
        
        with open(f"{output_path}_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save frequency tables
        self._save_frequency_tables(output_path, corpus_stats)
        
        self.logger.info(f"Statistics saved to {output_path}_*.{pkl,json,txt}")
    
    def _save_frequency_tables(self, output_path: Path, corpus_stats: CorpusStatistics):
        """Save frequency tables as text files"""
        # Character frequencies
        with open(f"{output_path}_char_freq.txt", 'w', encoding='utf-8') as f:
            f.write("Character Frequencies\n")
            f.write("====================\n\n")
            for char, freq in corpus_stats.character_stats.most_common_chars:
                f.write(f"{char}\t{freq}\n")
        
        # Syllable frequencies
        with open(f"{output_path}_syllable_freq.txt", 'w', encoding='utf-8') as f:
            f.write("Syllable Frequencies\n")
            f.write("===================\n\n")
            for syllable, freq in corpus_stats.syllable_stats.most_common_syllables:
                f.write(f"{syllable}\t{freq}\n")
        
        # Word frequencies
        with open(f"{output_path}_word_freq.txt", 'w', encoding='utf-8') as f:
            f.write("Word Frequencies\n")
            f.write("===============\n\n")
            for word, freq in corpus_stats.word_stats.most_common_words:
                f.write(f"{word}\t{freq}\n")
    
    def generate_report(self, corpus_stats: CorpusStatistics) -> str:
        """Generate human-readable statistical report"""
        report = []
        report.append("📊 KHMER CORPUS STATISTICAL ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Overview
        report.append("📋 CORPUS OVERVIEW")
        report.append("-" * 20)
        report.append(f"Total texts: {corpus_stats.total_texts:,}")
        report.append(f"Total size: {corpus_stats.total_size_bytes / (1024*1024):.1f} MB")
        report.append(f"Quality score: {corpus_stats.quality_score:.3f}")
        report.append("")
        
        # Character statistics
        cs = corpus_stats.character_stats
        report.append("🔤 CHARACTER ANALYSIS")
        report.append("-" * 20)
        report.append(f"Total characters: {cs.total_characters:,}")
        report.append(f"Unique characters: {cs.unique_characters:,}")
        report.append(f"Khmer ratio: {cs.khmer_ratio:.3f}")
        report.append(f"Top 5 characters: {', '.join([f'{c}({f})' for c, f in cs.most_common_chars[:5]])}")
        report.append("")
        
        # Syllable statistics
        ss = corpus_stats.syllable_stats
        report.append("🔣 SYLLABLE ANALYSIS")
        report.append("-" * 20)
        report.append(f"Total syllables: {ss.total_syllables:,}")
        report.append(f"Unique syllables: {ss.unique_syllables:,}")
        report.append(f"Average length: {ss.avg_syllable_length:.2f}")
        report.append(f"Top 5 syllables: {', '.join([f'{s}({f})' for s, f in ss.most_common_syllables[:5]])}")
        report.append("")
        
        # Word statistics
        ws = corpus_stats.word_stats
        report.append("📝 WORD ANALYSIS")
        report.append("-" * 20)
        report.append(f"Total words: {ws.total_words:,}")
        report.append(f"Unique words: {ws.unique_words:,}")
        report.append(f"Average length: {ws.avg_word_length:.2f}")
        report.append(f"Vocabulary coverage: {corpus_stats.vocabulary_coverage:.3f}")
        report.append(f"Top 5 words: {', '.join([f'{w}({f})' for w, f in ws.most_common_words[:5]])}")
        report.append("")
        
        # Quality metrics
        report.append("⭐ QUALITY METRICS")
        report.append("-" * 20)
        report.append(f"Text diversity: {corpus_stats.text_diversity:.3f}")
        report.append(f"Overall quality: {corpus_stats.quality_score:.3f}")
        report.append("")
        
        return "\n".join(report)


def analyze_corpus_statistics(texts: List[str], syllable_sequences: List[List[str]] = None) -> CorpusStatistics:
    """Convenience function to analyze corpus statistics"""
    analyzer = StatisticalAnalyzer()
    analyzer.analyze_corpus(texts, syllable_sequences)
    
    total_size = sum(len(text.encode('utf-8')) for text in texts)
    return analyzer.get_corpus_statistics(len(texts), total_size) 