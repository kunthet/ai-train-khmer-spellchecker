"""
Hybrid Validator for Khmer Spellchecking

This module combines rule-based validation with statistical n-gram models
to provide comprehensive spellchecking with both linguistic rules and
statistical probability assessment.
"""

import time
import logging
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path

try:
    from .rule_based_validator import RuleBasedValidator, ValidationResult as RuleValidationResult
    from .character_ngram_model import CharacterNgramModel, ErrorDetectionResult as CharErrorResult
    from .syllable_ngram_model import SyllableNgramModel, SyllableErrorDetectionResult as SyllableErrorResult
    from word_cluster.syllable_api import SyllableSegmentationAPI, SegmentationMethod
except ImportError:
    # Fallback for running as standalone script
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rule_based_validator import RuleBasedValidator, ValidationResult as RuleValidationResult
    from character_ngram_model import CharacterNgramModel, ErrorDetectionResult as CharErrorResult
    from syllable_ngram_model import SyllableNgramModel, SyllableErrorDetectionResult as SyllableErrorResult
    from word_cluster.syllable_api import SyllableSegmentationAPI, SegmentationMethod


@dataclass
class HybridError:
    """Combined error from multiple validation methods"""
    position: int
    length: int
    text_segment: str
    error_type: str
    confidence: float
    source: str  # 'rule', 'character', 'syllable'
    description: str
    suggestion: Optional[str] = None


@dataclass
class HybridValidationResult:
    """Result of hybrid validation combining multiple approaches"""
    text: str
    syllables: List[str]
    
    # Individual results
    rule_result: Optional[RuleValidationResult]
    character_result: Optional[CharErrorResult]
    syllable_result: Optional[SyllableErrorResult]
    
    # Combined analysis
    errors: List[HybridError]
    overall_score: float
    confidence_score: float
    
    # Method scores
    rule_score: float
    statistical_score: float
    consensus_score: float
    
    # Performance metrics
    processing_time: float
    
    @property
    def is_valid(self) -> bool:
        """Check if text is valid according to hybrid analysis"""
        return len(self.errors) == 0 and self.overall_score > 0.7


class HybridValidator:
    """
    Hybrid validator combining rule-based and statistical approaches
    
    Provides comprehensive Khmer text validation using:
    1. Rule-based linguistic validation
    2. Character-level n-gram statistical models
    3. Syllable-level n-gram statistical models
    """
    
    def __init__(self,
                 character_model_path: Optional[str] = None,
                 syllable_model_path: Optional[str] = None,
                 rule_validator: Optional[RuleBasedValidator] = None,
                 segmentation_method: SegmentationMethod = SegmentationMethod.REGEX_ADVANCED,
                 weights: Dict[str, float] = None):
        """
        Initialize hybrid validator
        
        Args:
            character_model_path: Path to trained character n-gram model
            syllable_model_path: Path to trained syllable n-gram model
            rule_validator: Pre-configured rule validator (optional)
            segmentation_method: Syllable segmentation method
            weights: Scoring weights for different methods
        """
        self.segmentation_method = segmentation_method
        self.segmentation_api = SyllableSegmentationAPI(segmentation_method)
        
        # Default weights for combining scores
        self.weights = weights or {
            'rule': 0.4,       # Rule-based validation weight
            'character': 0.3,  # Character n-gram weight  
            'syllable': 0.3    # Syllable n-gram weight
        }
        
        # Initialize validators
        self.rule_validator = rule_validator or RuleBasedValidator()
        self.character_model = None
        self.syllable_model = None
        
        # Load statistical models if paths provided
        if character_model_path:
            self.load_character_model(character_model_path)
        
        if syllable_model_path:
            self.load_syllable_model(syllable_model_path)
        
        self.logger = logging.getLogger("hybrid_validator")
        self.logger.info("Initialized hybrid validator")
        self.logger.info(f"Weights: rule={self.weights['rule']}, "
                        f"char={self.weights['character']}, "
                        f"syllable={self.weights['syllable']}")
    
    def load_character_model(self, model_path: str):
        """Load character n-gram model"""
        try:
            self.character_model = CharacterNgramModel(n=3)  # Default 3-gram
            self.character_model.load_model(model_path)
            self.logger.info(f"Loaded character model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load character model: {e}")
            self.character_model = None
    
    def load_syllable_model(self, model_path: str):
        """Load syllable n-gram model"""
        try:
            self.syllable_model = SyllableNgramModel(n=3)  # Default 3-gram
            self.syllable_model.load_model(model_path)
            self.logger.info(f"Loaded syllable model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load syllable model: {e}")
            self.syllable_model = None
    
    def validate_text(self, text: str) -> HybridValidationResult:
        """
        Perform comprehensive hybrid validation
        
        Args:
            text: Text to validate
            
        Returns:
            Hybrid validation result
        """
        start_time = time.time()
        
        # Segment text into syllables
        segmentation_result = self.segmentation_api.segment_text(text)
        syllables = segmentation_result.syllables if segmentation_result.success else []
        
        # Run individual validators
        rule_result = self._run_rule_validation(text, syllables)
        character_result = self._run_character_validation(text)
        syllable_result = self._run_syllable_validation(text)
        
        # Combine results
        combined_errors = self._combine_errors(rule_result, character_result, syllable_result)
        
        # Calculate scores
        rule_score = rule_result.overall_score if rule_result else 1.0
        
        char_score = 1.0 - (len(character_result.errors_detected) / max(len(text), 1)) if character_result else 1.0
        syllable_score = syllable_result.overall_score if syllable_result else 1.0
        
        statistical_score = (char_score + syllable_score) / 2
        
        # Weighted overall score
        overall_score = (
            self.weights['rule'] * rule_score +
            self.weights['character'] * char_score +
            self.weights['syllable'] * syllable_score
        )
        
        # Consensus score (agreement between methods)
        consensus_score = self._calculate_consensus(rule_result, character_result, syllable_result)
        
        # Confidence score based on agreement and model availability
        confidence_score = self._calculate_confidence(rule_result, character_result, syllable_result)
        
        processing_time = time.time() - start_time
        
        return HybridValidationResult(
            text=text,
            syllables=syllables,
            rule_result=rule_result,
            character_result=character_result,
            syllable_result=syllable_result,
            errors=combined_errors,
            overall_score=overall_score,
            confidence_score=confidence_score,
            rule_score=rule_score,
            statistical_score=statistical_score,
            consensus_score=consensus_score,
            processing_time=processing_time
        )
    
    def _run_rule_validation(self, text: str, syllables: List[str]) -> Optional[RuleValidationResult]:
        """Run rule-based validation"""
        try:
            return self.rule_validator.validate_text(text, syllables)
        except Exception as e:
            self.logger.error(f"Rule validation failed: {e}")
            return None
    
    def _run_character_validation(self, text: str) -> Optional[CharErrorResult]:
        """Run character n-gram validation"""
        if not self.character_model:
            return None
        
        try:
            return self.character_model.detect_errors(text)
        except Exception as e:
            self.logger.error(f"Character validation failed: {e}")
            return None
    
    def _run_syllable_validation(self, text: str) -> Optional[SyllableErrorResult]:
        """Run syllable n-gram validation"""
        if not self.syllable_model:
            return None
        
        try:
            return self.syllable_model.detect_errors(text)
        except Exception as e:
            self.logger.error(f"Syllable validation failed: {e}")
            return None
    
    def _combine_errors(self,
                       rule_result: Optional[RuleValidationResult],
                       character_result: Optional[CharErrorResult],
                       syllable_result: Optional[SyllableErrorResult]) -> List[HybridError]:
        """Combine errors from different validation methods"""
        combined_errors = []
        
        # Add rule-based errors
        if rule_result:
            for error in rule_result.errors:
                combined_errors.append(HybridError(
                    position=error.position,
                    length=error.length,
                    text_segment=error.text_segment,
                    error_type=error.error_type.value,
                    confidence=0.9,  # High confidence for rule-based errors
                    source='rule',
                    description=error.description,
                    suggestion=error.suggestion
                ))
        
        # Add character-level errors
        if character_result:
            for start, end, char_seq, confidence in character_result.errors_detected:
                combined_errors.append(HybridError(
                    position=start,
                    length=end - start,
                    text_segment=char_seq,
                    error_type='statistical_character',
                    confidence=confidence,
                    source='character',
                    description=f"Unusual character sequence: '{char_seq}'",
                    suggestion="Check character spelling"
                ))
        
        # Add syllable-level errors  
        if syllable_result:
            for start, end, syllable, confidence in syllable_result.errors_detected:
                combined_errors.append(HybridError(
                    position=start,
                    length=end - start,
                    text_segment=syllable,
                    error_type='statistical_syllable',
                    confidence=confidence,
                    source='syllable',
                    description=f"Unusual syllable: '{syllable}'",
                    suggestion="Check syllable spelling"
                ))
        
        # Remove duplicate errors (same position/segment)
        unique_errors = self._deduplicate_errors(combined_errors)
        
        # Sort by position
        unique_errors.sort(key=lambda e: e.position)
        
        return unique_errors
    
    def _deduplicate_errors(self, errors: List[HybridError]) -> List[HybridError]:
        """Remove duplicate errors from different sources"""
        seen_positions = {}
        unique_errors = []
        
        for error in errors:
            key = (error.position, error.text_segment)
            
            if key not in seen_positions:
                seen_positions[key] = error
                unique_errors.append(error)
            else:
                # If multiple sources detect same error, use highest confidence
                existing = seen_positions[key]
                if error.confidence > existing.confidence:
                    # Replace with higher confidence error
                    unique_errors = [e for e in unique_errors if e != existing]
                    unique_errors.append(error)
                    seen_positions[key] = error
        
        return unique_errors
    
    def _calculate_consensus(self,
                           rule_result: Optional[RuleValidationResult],
                           character_result: Optional[CharErrorResult],
                           syllable_result: Optional[SyllableErrorResult]) -> float:
        """Calculate consensus score between different methods"""
        scores = []
        
        if rule_result:
            scores.append(rule_result.overall_score)
        
        if character_result:
            char_score = 1.0 - (len(character_result.errors_detected) / max(len(character_result.text), 1))
            scores.append(char_score)
        
        if syllable_result:
            scores.append(syllable_result.overall_score)
        
        if not scores:
            return 0.0
        
        # Calculate variance in scores (lower variance = higher consensus)
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Convert variance to consensus score (0-1, higher is better)
        consensus = max(0.0, 1.0 - variance * 4)  # Scale variance
        
        return consensus
    
    def _calculate_confidence(self,
                            rule_result: Optional[RuleValidationResult],
                            character_result: Optional[CharErrorResult],
                            syllable_result: Optional[SyllableErrorResult]) -> float:
        """Calculate confidence score based on model availability and agreement"""
        available_methods = 0
        
        if rule_result:
            available_methods += 1
        if character_result:
            available_methods += 1
        if syllable_result:
            available_methods += 1
        
        # Base confidence on number of available methods
        method_confidence = available_methods / 3.0
        
        # Boost confidence if methods agree
        consensus = self._calculate_consensus(rule_result, character_result, syllable_result)
        
        # Combined confidence
        confidence = (method_confidence + consensus) / 2.0
        
        return confidence
    
    def validate_batch(self, texts: List[str]) -> List[HybridValidationResult]:
        """Validate multiple texts"""
        results = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                self.logger.info(f"Processing text {i+1}/{len(texts)}")
            
            result = self.validate_text(text)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[HybridValidationResult]) -> str:
        """Generate comprehensive validation report"""
        if not results:
            return "No validation results to report"
        
        report = []
        report.append("🔄 HYBRID VALIDATION REPORT")
        report.append("=" * 40)
        report.append("")
        
        # Overall statistics
        total_texts = len(results)
        valid_texts = sum(1 for r in results if r.is_valid)
        total_errors = sum(len(r.errors) for r in results)
        avg_overall_score = sum(r.overall_score for r in results) / total_texts
        avg_confidence = sum(r.confidence_score for r in results) / total_texts
        avg_consensus = sum(r.consensus_score for r in results) / total_texts
        
        report.append("📊 OVERALL STATISTICS")
        report.append("-" * 25)
        report.append(f"Total texts validated: {total_texts:,}")
        report.append(f"Valid texts: {valid_texts:,} ({valid_texts/total_texts:.1%})")
        report.append(f"Total errors detected: {total_errors:,}")
        report.append(f"Average overall score: {avg_overall_score:.3f}")
        report.append(f"Average confidence: {avg_confidence:.3f}")
        report.append(f"Average consensus: {avg_consensus:.3f}")
        report.append("")
        
        # Method performance
        rule_scores = [r.rule_score for r in results if r.rule_result]
        statistical_scores = [r.statistical_score for r in results]
        
        report.append("🎯 METHOD PERFORMANCE")
        report.append("-" * 25)
        if rule_scores:
            report.append(f"Rule-based average: {sum(rule_scores)/len(rule_scores):.3f}")
        report.append(f"Statistical average: {sum(statistical_scores)/len(statistical_scores):.3f}")
        report.append("")
        
        # Error source breakdown
        error_sources = {}
        error_types = {}
        
        for result in results:
            for error in result.errors:
                error_sources[error.source] = error_sources.get(error.source, 0) + 1
                error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        if error_sources:
            report.append("🔍 ERROR SOURCE BREAKDOWN")
            report.append("-" * 30)
            for source, count in sorted(error_sources.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_errors) * 100
                report.append(f"{source}: {count:,} ({percentage:.1f}%)")
            report.append("")
        
        if error_types:
            report.append("📝 ERROR TYPE BREAKDOWN")
            report.append("-" * 25)
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_errors) * 100
                report.append(f"{error_type}: {count:,} ({percentage:.1f}%)")
            report.append("")
        
        # Performance metrics
        avg_processing_time = sum(r.processing_time for r in results) / total_texts
        total_processing_time = sum(r.processing_time for r in results)
        
        report.append("⚡ PERFORMANCE METRICS")
        report.append("-" * 25)
        report.append(f"Average processing time: {avg_processing_time*1000:.1f}ms")
        report.append(f"Total processing time: {total_processing_time:.2f}s")
        
        # Avoid division by zero for processing rate
        if total_processing_time > 0:
            processing_rate = total_texts / total_processing_time
            report.append(f"Processing rate: {processing_rate:.1f} texts/second")
        else:
            report.append(f"Processing rate: Very fast (too quick to measure accurately)")
        
        report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Demo usage
    print("🔄 HYBRID VALIDATOR DEMO")
    print("=" * 30)
    
    # Initialize hybrid validator (without pre-trained models for demo)
    validator = HybridValidator()
    
    # Sample test texts
    test_texts = [
        "នេះជាការសាកល្បងត្រឹមត្រូវ។",  # Correct text
        "នេះជាការសាកល្បងខុស។",  # Text with potential issues
        "កំ្",  # Invalid coeng usage
        "្ក",  # Coeng at beginning
        "Hello និង ខ្មែរ",  # Mixed language
    ]
    
    print("Testing hybrid validation...")
    results = []
    
    for i, text in enumerate(test_texts, 1):
        result = validator.validate_text(text)
        results.append(result)
        
        print(f"\nTest {i}: '{text}'")
        print(f"Valid: {result.is_valid}")
        print(f"Overall score: {result.overall_score:.3f}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Consensus: {result.consensus_score:.3f}")
        print(f"Errors detected: {len(result.errors)}")
        
        if result.errors:
            print("Error details:")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"  - [{error.source}] {error.description}")
    
    # Generate report
    print(f"\n{validator.generate_report(results)}")
    print("Hybrid validation demo completed!") 