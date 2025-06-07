"""
Syllable Segmentation API

This module provides a unified, standardized interface for Khmer syllable segmentation
with enhanced error handling, performance monitoring, and batch processing capabilities.
"""

import logging
import time
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from word_cluster.subword_cluster import (
    khmer_syllables_no_regex_fast,
    khmer_syllables_no_regex,
    khmer_syllables,
    khmer_syllables_advanced
)


class SegmentationMethod(Enum):
    """Available syllable segmentation methods"""
    NO_REGEX_FAST = "no_regex_fast"
    NO_REGEX = "no_regex"
    REGEX_BASIC = "regex_basic"
    REGEX_ADVANCED = "regex_advanced"


@dataclass
class SegmentationResult:
    """Result of syllable segmentation operation"""
    text: str
    syllables: List[str]
    method: str
    processing_time: float
    syllable_count: int
    character_count: int
    success: bool = True
    error_message: str = ""
    
    def __post_init__(self):
        if self.success:
            self.syllable_count = len(self.syllables)
            self.character_count = len(self.text)


@dataclass
class BatchSegmentationResult:
    """Result of batch syllable segmentation"""
    total_texts: int
    successful_texts: int
    failed_texts: int
    total_syllables: int
    total_characters: int
    total_processing_time: float
    avg_processing_time: float
    results: List[SegmentationResult]
    method: str
    
    def __post_init__(self):
        if self.total_texts > 0:
            self.avg_processing_time = self.total_processing_time / self.total_texts


class SyllableSegmentationAPI:
    """Unified API for Khmer syllable segmentation"""
    
    def __init__(self, default_method: SegmentationMethod = SegmentationMethod.REGEX_ADVANCED):
        self.default_method = default_method
        self.logger = logging.getLogger("syllable_api")
        
        # Method mapping
        self._methods = {
            SegmentationMethod.NO_REGEX_FAST: khmer_syllables_no_regex_fast,
            SegmentationMethod.NO_REGEX: khmer_syllables_no_regex,
            SegmentationMethod.REGEX_BASIC: khmer_syllables,
            SegmentationMethod.REGEX_ADVANCED: khmer_syllables_advanced
        }
        
        # Performance statistics
        self.stats = {
            'total_texts_processed': 0,
            'total_syllables_generated': 0,
            'total_processing_time': 0.0,
            'successful_operations': 0,
            'failed_operations': 0,
            'method_usage': {method.value: 0 for method in SegmentationMethod}
        }
        
        self.logger.info(f"SyllableSegmentationAPI initialized with default method: {default_method.value}")
    
    def segment_text(
        self, 
        text: str, 
        method: Optional[SegmentationMethod] = None
    ) -> SegmentationResult:
        """
        Segment a single text into syllables
        
        Args:
            text: Input text to segment
            method: Segmentation method to use (defaults to instance default)
            
        Returns:
            SegmentationResult with syllables and metadata
        """
        if method is None:
            method = self.default_method
        
        # Input validation
        if not isinstance(text, str):
            return SegmentationResult(
                text=str(text),
                syllables=[],
                method=method.value,
                processing_time=0.0,
                syllable_count=0,
                character_count=0,
                success=False,
                error_message="Input must be a string"
            )
        
        if not text.strip():
            return SegmentationResult(
                text=text,
                syllables=[],
                method=method.value,
                processing_time=0.0,
                syllable_count=0,
                character_count=len(text),
                success=True
            )
        
        # Get segmentation function
        segment_func = self._methods.get(method)
        if not segment_func:
            return SegmentationResult(
                text=text,
                syllables=[],
                method=method.value,
                processing_time=0.0,
                syllable_count=0,
                character_count=len(text),
                success=False,
                error_message=f"Unknown segmentation method: {method.value}"
            )
        
        # Perform segmentation with timing
        start_time = time.time()
        try:
            syllables = segment_func(text)
            processing_time = time.time() - start_time
            
            # Validate result
            if not isinstance(syllables, list):
                raise ValueError(f"Segmentation function returned {type(syllables)}, expected list")
            
            # Update statistics
            self.stats['total_texts_processed'] += 1
            self.stats['total_syllables_generated'] += len(syllables)
            self.stats['total_processing_time'] += processing_time
            self.stats['successful_operations'] += 1
            self.stats['method_usage'][method.value] += 1
            
            return SegmentationResult(
                text=text,
                syllables=syllables,
                method=method.value,
                processing_time=processing_time,
                syllable_count=len(syllables),
                character_count=len(text),
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update error statistics
            self.stats['failed_operations'] += 1
            self.stats['total_processing_time'] += processing_time
            
            error_msg = f"Segmentation failed: {str(e)}"
            self.logger.error(f"Error segmenting text with {method.value}: {error_msg}")
            
            return SegmentationResult(
                text=text,
                syllables=[],
                method=method.value,
                processing_time=processing_time,
                syllable_count=0,
                character_count=len(text),
                success=False,
                error_message=error_msg
            )
    
    def segment_batch(
        self, 
        texts: List[str], 
        method: Optional[SegmentationMethod] = None,
        show_progress: bool = True
    ) -> BatchSegmentationResult:
        """
        Segment multiple texts in batch
        
        Args:
            texts: List of texts to segment
            method: Segmentation method to use
            show_progress: Whether to log progress updates
            
        Returns:
            BatchSegmentationResult with aggregated statistics
        """
        if method is None:
            method = self.default_method
        
        results = []
        total_start_time = time.time()
        
        for i, text in enumerate(texts):
            result = self.segment_text(text, method)
            results.append(result)
            
            # Progress logging
            if show_progress and (i + 1) % 1000 == 0:
                self.logger.info(f"Processed {i + 1}/{len(texts)} texts with {method.value}")
        
        total_processing_time = time.time() - total_start_time
        
        # Calculate aggregated statistics
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        total_syllables = sum(r.syllable_count for r in successful_results)
        total_characters = sum(r.character_count for r in results)
        
        return BatchSegmentationResult(
            total_texts=len(texts),
            successful_texts=len(successful_results),
            failed_texts=len(failed_results),
            total_syllables=total_syllables,
            total_characters=total_characters,
            total_processing_time=total_processing_time,
            avg_processing_time=total_processing_time / len(texts) if texts else 0,
            results=results,
            method=method.value
        )
    
    def compare_methods(
        self, 
        text: str, 
        methods: Optional[List[SegmentationMethod]] = None
    ) -> Dict[str, SegmentationResult]:
        """
        Compare multiple segmentation methods on the same text
        
        Args:
            text: Text to segment with all methods
            methods: List of methods to compare (defaults to all methods)
            
        Returns:
            Dictionary mapping method names to results
        """
        if methods is None:
            methods = list(SegmentationMethod)
        
        results = {}
        
        for method in methods:
            result = self.segment_text(text, method)
            results[method.value] = result
        
        return results
    
    def validate_consistency(
        self, 
        text: str, 
        methods: Optional[List[SegmentationMethod]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that all methods produce consistent results
        
        Args:
            text: Text to validate consistency for
            methods: Methods to compare (defaults to all)
            
        Returns:
            Tuple of (is_consistent, detailed_results)
        """
        results = self.compare_methods(text, methods)
        
        # Extract syllable lists for comparison
        syllable_lists = {}
        for method_name, result in results.items():
            if result.success:
                syllable_lists[method_name] = result.syllables
        
        # Check consistency
        if not syllable_lists:
            return False, {"error": "No successful segmentations to compare"}
        
        # Compare all results to the first one
        first_result = list(syllable_lists.values())[0]
        is_consistent = all(syllables == first_result for syllables in syllable_lists.values())
        
        return is_consistent, {
            "consistent": is_consistent,
            "results": results,
            "syllable_lists": syllable_lists,
            "first_result": first_result,
            "comparison": {
                method: syllables == first_result 
                for method, syllables in syllable_lists.items()
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current API usage statistics"""
        total_operations = self.stats['successful_operations'] + self.stats['failed_operations']
        
        return {
            **self.stats,
            'total_operations': total_operations,
            'success_rate': self.stats['successful_operations'] / total_operations if total_operations > 0 else 0,
            'avg_processing_time': self.stats['total_processing_time'] / total_operations if total_operations > 0 else 0,
            'avg_syllables_per_text': self.stats['total_syllables_generated'] / self.stats['successful_operations'] if self.stats['successful_operations'] > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset all performance statistics"""
        self.stats = {
            'total_texts_processed': 0,
            'total_syllables_generated': 0,
            'total_processing_time': 0.0,
            'successful_operations': 0,
            'failed_operations': 0,
            'method_usage': {method.value: 0 for method in SegmentationMethod}
        }
        self.logger.info("Statistics reset")


# Convenience functions for backward compatibility
def segment_text(text: str, method: str = "regex_advanced") -> List[str]:
    """Simple function to segment text using specified method"""
    api = SyllableSegmentationAPI()
    method_enum = SegmentationMethod(method)
    result = api.segment_text(text, method_enum)
    
    if result.success:
        return result.syllables
    else:
        raise ValueError(f"Segmentation failed: {result.error_message}")


def segment_texts(texts: List[str], method: str = "regex_advanced") -> List[List[str]]:
    """Simple function to segment multiple texts"""
    api = SyllableSegmentationAPI()
    method_enum = SegmentationMethod(method)
    batch_result = api.segment_batch(texts, method_enum, show_progress=False)
    
    return [result.syllables for result in batch_result.results if result.success]


if __name__ == "__main__":
    # Demo usage
    api = SyllableSegmentationAPI(SegmentationMethod.REGEX_ADVANCED)
    
    # Test single text
    test_text = "នេះជាការសាកល្បងអត្ថបទខ្មែរ។"
    result = api.segment_text(test_text)
    print(f"Text: {test_text}")
    print(f"Syllables: {result.syllables}")
    print(f"Success: {result.success}")
    print(f"Processing time: {result.processing_time:.4f}s")
    
    # Test method comparison
    print("\n=== Method Comparison ===")
    comparison = api.compare_methods(test_text)
    for method, result in comparison.items():
        print(f"{method}: {result.syllables} ({result.processing_time:.4f}s)")
    
    # Test consistency validation
    print("\n=== Consistency Validation ===")
    is_consistent, details = api.validate_consistency(test_text)
    print(f"All methods consistent: {is_consistent}")
    
    # Show statistics
    print("\n=== API Statistics ===")
    stats = api.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}") 