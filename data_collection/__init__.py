"""
Khmer Text Data Collection Package

This package provides tools for collecting Khmer text data from various sources
including news websites, Wikipedia, and government documents.
"""

__version__ = "1.0.0"
__author__ = "Khmer Spellchecker Development Team"

from .web_scrapers import (
    VOAKhmerScraper,
    KhmerTimesScraper,
    RFAKhmerScraper,
    BaseScraper,
    create_scraper
)

from .text_processor import TextProcessor
from .deduplicator import ContentDeduplicator
from .file_loader import FileLoader, TextDocument, load_khmer_corpus, get_corpus_preview

__all__ = [
    'VOAKhmerScraper',
    'KhmerTimesScraper', 
    'RFAKhmerScraper',
    'BaseScraper',
    'create_scraper',
    'TextProcessor',
    'ContentDeduplicator',
    'FileLoader',
    'TextDocument',
    'load_khmer_corpus',
    'get_corpus_preview'
] 