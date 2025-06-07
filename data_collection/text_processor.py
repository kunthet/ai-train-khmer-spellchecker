"""
Text Processing Module

This module provides tools for cleaning, normalizing, and validating
Khmer text content extracted from web sources.
"""

import re
import unicodedata
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TextQualityMetrics:
    """Metrics for text quality assessment"""
    khmer_ratio: float
    length: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    has_proper_encoding: bool
    quality_score: float


class TextProcessor:
    """Advanced text processing and quality validation"""
    
    def __init__(
        self,
        min_khmer_ratio: float = 0.6,
        min_length: int = 50,
        max_english_ratio: float = 0.3,
        remove_brackets: bool = True,
        normalize_unicode: bool = True
    ):
        self.min_khmer_ratio = min_khmer_ratio
        self.min_length = min_length
        self.max_english_ratio = max_english_ratio
        self.remove_brackets = remove_brackets
        self.normalize_unicode = normalize_unicode
        
        self.logger = logging.getLogger("text_processor")
        
        # Statistics
        self.stats = {
            'texts_processed': 0,
            'texts_accepted': 0,
            'texts_rejected_length': 0,
            'texts_rejected_khmer_ratio': 0,
            'texts_rejected_encoding': 0,
            'texts_rejected_quality': 0
        }
    
    def clean_html_artifacts(self, text: str) -> str:
        """Remove HTML artifacts and markup"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        text = re.sub(r'&#\d+;', '', text)
        
        # Remove common web artifacts
        text = re.sub(r'javascript:.*?;', '', text)
        text = re.sub(r'mailto:.*?[\s>]', '', text)
        
        return text.strip()
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and remove extra spaces"""
        if not text:
            return ""
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text
    
    def remove_unwanted_content(self, text: str) -> str:
        """Remove unwanted content patterns"""
        if not text:
            return ""
        
        if self.remove_brackets:
            # Remove bracketed content
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\(.*?\)', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove phone numbers (common patterns)
        text = re.sub(r'\+?[\d\s\-\(\)]{10,}', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[\?]{2,}', '?', text)
        
        return text.strip()
    
    def normalize_unicode_text(self, text: str) -> str:
        """Normalize Unicode text"""
        if not text:
            return ""
        
        if self.normalize_unicode:
            # Normalize Unicode to NFC form
            text = unicodedata.normalize('NFC', text)
        
        # Remove zero-width characters
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\ufeff', '')  # Byte order mark
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        
        return text
    
    def calculate_khmer_ratio(self, text: str) -> float:
        """Calculate ratio of Khmer characters in text"""
        if not text:
            return 0.0
        
        khmer_chars = sum(1 for char in text if '\u1780' <= char <= '\u17FF')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return 0.0
        
        return khmer_chars / total_chars
    
    def calculate_english_ratio(self, text: str) -> float:
        """Calculate ratio of English characters in text"""
        if not text:
            return 0.0
        
        english_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return 0.0
        
        return english_chars / total_chars
    
    def validate_encoding(self, text: str) -> bool:
        """Validate that text has proper UTF-8 encoding"""
        try:
            # Try to encode and decode
            text.encode('utf-8').decode('utf-8')
            
            # Check for common encoding issues
            if '�' in text:  # Replacement character
                return False
            
            # Check for malformed Khmer sequences
            # (This is a basic check - could be expanded)
            if re.search(r'[\u1780-\u17FF]{20,}', text):  # Too many consecutive Khmer chars
                return False
            
            return True
        except (UnicodeEncodeError, UnicodeDecodeError):
            return False
    
    def calculate_quality_metrics(self, text: str) -> TextQualityMetrics:
        """Calculate comprehensive text quality metrics"""
        if not text:
            return TextQualityMetrics(
                khmer_ratio=0.0,
                length=0,
                word_count=0,
                sentence_count=0,
                avg_word_length=0.0,
                has_proper_encoding=False,
                quality_score=0.0
            )
        
        # Basic metrics
        length = len(text)
        khmer_ratio = self.calculate_khmer_ratio(text)
        has_proper_encoding = self.validate_encoding(text)
        
        # Word and sentence counting
        words = text.split()
        word_count = len(words)
        
        # Count sentences (basic approach)
        sentence_count = len(re.findall(r'[.!?]+', text))
        if sentence_count == 0:
            sentence_count = 1
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            khmer_ratio, length, word_count, sentence_count, 
            avg_word_length, has_proper_encoding
        )
        
        return TextQualityMetrics(
            khmer_ratio=khmer_ratio,
            length=length,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            has_proper_encoding=has_proper_encoding,
            quality_score=quality_score
        )
    
    def _calculate_quality_score(
        self, 
        khmer_ratio: float, 
        length: int, 
        word_count: int, 
        sentence_count: int,
        avg_word_length: float, 
        has_proper_encoding: bool
    ) -> float:
        """Calculate overall quality score (0-1)"""
        score = 0.0
        
        # Khmer ratio score (30% weight)
        if khmer_ratio >= self.min_khmer_ratio:
            score += 0.3 * min(khmer_ratio / 0.8, 1.0)
        
        # Length score (20% weight)
        if length >= self.min_length:
            score += 0.2 * min(length / 500, 1.0)
        
        # Word density score (20% weight)
        if word_count > 0:
            word_density = word_count / max(length, 1)
            score += 0.2 * min(word_density * 10, 1.0)
        
        # Sentence structure score (15% weight)
        if sentence_count > 0:
            words_per_sentence = word_count / sentence_count
            if 5 <= words_per_sentence <= 25:  # Reasonable range
                score += 0.15
        
        # Average word length score (10% weight)
        if 3 <= avg_word_length <= 15:  # Reasonable range for Khmer
            score += 0.1
        
        # Encoding score (5% weight)
        if has_proper_encoding:
            score += 0.05
        
        return min(score, 1.0)
    
    def is_high_quality(self, text: str, min_quality_score: float = 0.7) -> bool:
        """Check if text meets quality standards"""
        metrics = self.calculate_quality_metrics(text)
        return metrics.quality_score >= min_quality_score
    
    def clean_text(self, text: str) -> str:
        """Apply all cleaning operations"""
        if not text:
            return ""
        
        # Apply cleaning steps in order
        text = self.clean_html_artifacts(text)
        text = self.normalize_unicode_text(text)
        text = self.remove_unwanted_content(text)
        text = self.normalize_whitespace(text)
        
        return text
    
    def process_and_validate(self, text: str) -> Optional[str]:
        """Process text and validate quality"""
        self.stats['texts_processed'] += 1
        
        if not text:
            self.stats['texts_rejected_length'] += 1
            return None
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Calculate metrics
        metrics = self.calculate_quality_metrics(cleaned_text)
        
        # Validation checks
        if metrics.length < self.min_length:
            self.stats['texts_rejected_length'] += 1
            self.logger.debug(f"Rejected: too short ({metrics.length} chars)")
            return None
        
        if metrics.khmer_ratio < self.min_khmer_ratio:
            self.stats['texts_rejected_khmer_ratio'] += 1
            self.logger.debug(f"Rejected: low Khmer ratio ({metrics.khmer_ratio:.3f})")
            return None
        
        if not metrics.has_proper_encoding:
            self.stats['texts_rejected_encoding'] += 1
            self.logger.debug("Rejected: encoding issues")
            return None
        
        english_ratio = self.calculate_english_ratio(cleaned_text)
        if english_ratio > self.max_english_ratio:
            self.stats['texts_rejected_khmer_ratio'] += 1
            self.logger.debug(f"Rejected: too much English ({english_ratio:.3f})")
            return None
        
        if metrics.quality_score < 0.5:  # Basic quality threshold
            self.stats['texts_rejected_quality'] += 1
            self.logger.debug(f"Rejected: low quality score ({metrics.quality_score:.3f})")
            return None
        
        self.stats['texts_accepted'] += 1
        return cleaned_text
    
    def process_articles(self, articles: List) -> List:
        """Process a list of articles"""
        self.logger.info(f"Processing {len(articles)} articles")
        
        processed_articles = []
        
        for i, article in enumerate(articles):
            # Process title
            clean_title = self.clean_text(article.title)
            
            # Process content
            clean_content = self.process_and_validate(article.content)
            
            if clean_content:
                # Update article with cleaned content
                article.title = clean_title
                article.content = clean_content
                processed_articles.append(article)
            
            # Progress update
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i+1}/{len(articles)} articles")
        
        self.logger.info(f"Processing completed: {len(processed_articles)}/{len(articles)} articles kept")
        self._log_statistics()
        
        return processed_articles
    
    def _log_statistics(self):
        """Log processing statistics"""
        self.logger.info("=== Text Processing Statistics ===")
        for key, value in self.stats.items():
            self.logger.info(f"{key}: {value}")
        
        if self.stats['texts_processed'] > 0:
            acceptance_rate = self.stats['texts_accepted'] / self.stats['texts_processed']
            self.logger.info(f"acceptance_rate: {acceptance_rate:.3f}")


def clean_khmer_text(text: str, **kwargs) -> str:
    """Convenience function for text cleaning"""
    processor = TextProcessor(**kwargs)
    return processor.clean_text(text)


def validate_khmer_text(text: str, **kwargs) -> bool:
    """Convenience function for text validation"""
    processor = TextProcessor(**kwargs)
    return processor.process_and_validate(text) is not None 