"""
Content Deduplication Module

This module provides tools for detecting and removing duplicate content
across different scraped articles using various similarity metrics.
"""

import hashlib
import re
from typing import List, Set, Dict, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
import logging


@dataclass
class DuplicationResult:
    """Result of duplication analysis"""
    is_duplicate: bool
    similarity_score: float
    duplicate_type: str
    original_index: int = -1


class ContentDeduplicator:
    """Advanced content deduplication system"""
    
    def __init__(
        self,
        exact_match_threshold: float = 1.0,
        fuzzy_match_threshold: float = 0.85,
        title_similarity_threshold: float = 0.9,
        min_content_length: int = 50
    ):
        self.exact_threshold = exact_match_threshold
        self.fuzzy_threshold = fuzzy_match_threshold
        self.title_threshold = title_similarity_threshold
        self.min_length = min_content_length
        
        # Storage for hashes and processed content
        self.exact_hashes: Set[str] = set()
        self.fuzzy_hashes: Dict[str, int] = {}
        self.title_hashes: Dict[str, int] = {}
        self.processed_articles: List = []
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'exact_duplicates': 0,
            'fuzzy_duplicates': 0,
            'title_duplicates': 0,
            'short_content_removed': 0
        }
        
        self.logger = logging.getLogger("deduplicator")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove punctuation for fuzzy matching
        text = re.sub(r'[^\w\s\u1780-\u17FF]', '', text)
        
        return text
    
    def _get_exact_hash(self, content: str) -> str:
        """Generate exact content hash"""
        normalized = self._normalize_text(content)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _get_fuzzy_hash(self, content: str, chunk_size: int = 50) -> str:
        """Generate fuzzy hash for near-duplicate detection"""
        normalized = self._normalize_text(content)
        
        # Create chunks and hash them
        chunks = [normalized[i:i+chunk_size] for i in range(0, len(normalized), chunk_size)]
        chunk_hashes = [hashlib.md5(chunk.encode('utf-8')).hexdigest()[:8] for chunk in chunks]
        
        # Combine chunk hashes
        combined = ''.join(sorted(chunk_hashes))
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def _is_exact_duplicate(self, content: str) -> bool:
        """Check for exact duplicate"""
        content_hash = self._get_exact_hash(content)
        
        if content_hash in self.exact_hashes:
            return True
        
        self.exact_hashes.add(content_hash)
        return False
    
    def _is_fuzzy_duplicate(self, content: str, article_index: int) -> Tuple[bool, int]:
        """Check for fuzzy duplicate"""
        fuzzy_hash = self._get_fuzzy_hash(content)
        
        # Check against existing fuzzy hashes
        for existing_hash, existing_index in self.fuzzy_hashes.items():
            # Compare the actual content for similarity
            existing_content = self.processed_articles[existing_index].content
            similarity = self._calculate_similarity(content, existing_content)
            
            if similarity >= self.fuzzy_threshold:
                return True, existing_index
        
        self.fuzzy_hashes[fuzzy_hash] = article_index
        return False, -1
    
    def _is_title_duplicate(self, title: str, article_index: int) -> Tuple[bool, int]:
        """Check for title duplicate"""
        title_hash = self._get_exact_hash(title)
        
        # Check for exact title match first
        if title_hash in self.title_hashes:
            return True, self.title_hashes[title_hash]
        
        # Check for fuzzy title match
        for existing_index in range(len(self.processed_articles)):
            existing_title = self.processed_articles[existing_index].title
            similarity = self._calculate_similarity(title, existing_title)
            
            if similarity >= self.title_threshold:
                return True, existing_index
        
        self.title_hashes[title_hash] = article_index
        return False, -1
    
    def check_duplicate(self, article, current_index: int) -> DuplicationResult:
        """Check if article is duplicate"""
        content = article.content
        title = article.title
        
        # Check content length
        if len(content) < self.min_length:
            self.stats['short_content_removed'] += 1
            return DuplicationResult(
                is_duplicate=True,
                similarity_score=0.0,
                duplicate_type="short_content"
            )
        
        # Check exact duplicate
        if self._is_exact_duplicate(content):
            self.stats['exact_duplicates'] += 1
            return DuplicationResult(
                is_duplicate=True,
                similarity_score=1.0,
                duplicate_type="exact_content"
            )
        
        # Check fuzzy duplicate
        is_fuzzy_dup, fuzzy_index = self._is_fuzzy_duplicate(content, current_index)
        if is_fuzzy_dup:
            similarity = self._calculate_similarity(
                content, 
                self.processed_articles[fuzzy_index].content
            )
            self.stats['fuzzy_duplicates'] += 1
            return DuplicationResult(
                is_duplicate=True,
                similarity_score=similarity,
                duplicate_type="fuzzy_content",
                original_index=fuzzy_index
            )
        
        # Check title duplicate
        is_title_dup, title_index = self._is_title_duplicate(title, current_index)
        if is_title_dup:
            similarity = self._calculate_similarity(title, self.processed_articles[title_index].title)
            self.stats['title_duplicates'] += 1
            return DuplicationResult(
                is_duplicate=True,
                similarity_score=similarity,
                duplicate_type="title",
                original_index=title_index
            )
        
        return DuplicationResult(
            is_duplicate=False,
            similarity_score=0.0,
            duplicate_type="unique"
        )
    
    def deduplicate_articles(self, articles: List) -> List:
        """Remove duplicates from article list"""
        self.logger.info(f"Starting deduplication of {len(articles)} articles")
        
        unique_articles = []
        self.processed_articles = []
        
        for i, article in enumerate(articles):
            self.stats['total_processed'] += 1
            
            # Check for duplicates
            dup_result = self.check_duplicate(article, i)
            
            if not dup_result.is_duplicate:
                unique_articles.append(article)
                self.processed_articles.append(article)
                self.logger.debug(f"Article {i+1}: Unique - keeping")
            else:
                self.logger.debug(
                    f"Article {i+1}: {dup_result.duplicate_type} duplicate "
                    f"(similarity: {dup_result.similarity_score:.3f}) - removing"
                )
            
            # Progress update
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i+1}/{len(articles)} articles")
        
        self.logger.info(f"Deduplication completed: {len(unique_articles)}/{len(articles)} articles kept")
        self._log_statistics()
        
        return unique_articles
    
    def _log_statistics(self):
        """Log deduplication statistics"""
        self.logger.info("=== Deduplication Statistics ===")
        for key, value in self.stats.items():
            self.logger.info(f"{key}: {value}")
        
        if self.stats['total_processed'] > 0:
            unique_ratio = (self.stats['total_processed'] - 
                           self.stats['exact_duplicates'] - 
                           self.stats['fuzzy_duplicates'] - 
                           self.stats['title_duplicates'] - 
                           self.stats['short_content_removed']) / self.stats['total_processed']
            self.logger.info(f"unique_ratio: {unique_ratio:.3f}")
    
    def reset(self):
        """Reset deduplicator state"""
        self.exact_hashes.clear()
        self.fuzzy_hashes.clear()
        self.title_hashes.clear()
        self.processed_articles.clear()
        
        for key in self.stats:
            self.stats[key] = 0
        
        self.logger.info("Deduplicator state reset")


def remove_duplicates(articles: List, **kwargs) -> List:
    """Convenience function for deduplication"""
    deduplicator = ContentDeduplicator(**kwargs)
    return deduplicator.deduplicate_articles(articles) 