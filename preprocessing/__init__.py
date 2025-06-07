"""
Khmer Text Preprocessing Pipeline

This package provides advanced preprocessing capabilities for Khmer text,
including syllable segmentation integration, statistical analysis, and
quality validation specifically designed for spellchecker development.
"""

__version__ = "1.0.0"
__author__ = "Khmer Spellchecker Development Team"

from .text_pipeline import (
    TextPreprocessingPipeline,
    ProcessingResult,
    CorpusProcessor
)

from .statistical_analyzer import (
    StatisticalAnalyzer,
    CharacterStatistics,
    WordStatistics,
    CorpusStatistics
)

__all__ = [
    'TextPreprocessingPipeline',
    'ProcessingResult',
    'CorpusProcessor',
    'StatisticalAnalyzer',
    'CharacterStatistics',
    'WordStatistics',
    'CorpusStatistics'
] 