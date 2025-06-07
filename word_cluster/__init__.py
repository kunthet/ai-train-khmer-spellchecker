"""
Khmer Word Clustering and Syllable Segmentation Package

This package provides comprehensive tools for Khmer text processing,
including syllable segmentation, frequency analysis, and validation.
"""

__version__ = "1.0.0"
__author__ = "Khmer Spellchecker Development Team"

from .subword_cluster import (
    khmer_syllables_no_regex_fast,
    khmer_syllables_no_regex,
    khmer_syllables,
    khmer_syllables_advanced
)

from .syllable_api import (
    SyllableSegmentationAPI,
    SegmentationMethod,
    SegmentationResult,
    BatchSegmentationResult,
    segment_text,
    segment_texts
)

from .syllable_frequency_analyzer import (
    SyllableFrequencyAnalyzer,
    SyllableFrequencyStats,
    SyllableValidationResult
)

__all__ = [
    # Core segmentation functions
    'khmer_syllables_no_regex_fast',
    'khmer_syllables_no_regex',
    'khmer_syllables',
    'khmer_syllables_advanced',
    
    # Enhanced API
    'SyllableSegmentationAPI',
    'SegmentationMethod',
    'SegmentationResult',
    'BatchSegmentationResult',
    'segment_text',
    'segment_texts',
    
    # Frequency analysis
    'SyllableFrequencyAnalyzer',
    'SyllableFrequencyStats',
    'SyllableValidationResult'
] 