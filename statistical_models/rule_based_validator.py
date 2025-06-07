"""
Rule-Based Validation for Khmer Text

This module provides comprehensive rule-based validation for Khmer script,
including syllable structure validation, Unicode sequence checking, and
diacritic combination rules.
"""

import re
import logging
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationErrorType(Enum):
    """Types of validation errors"""
    INVALID_SYLLABLE_STRUCTURE = "invalid_syllable_structure"
    INVALID_UNICODE_SEQUENCE = "invalid_unicode_sequence"
    INVALID_COENG_USAGE = "invalid_coeng_usage"
    INVALID_VOWEL_COMBINATION = "invalid_vowel_combination"
    INVALID_DIACRITIC_PLACEMENT = "invalid_diacritic_placement"
    ORPHANED_COMBINING_MARK = "orphaned_combining_mark"
    INVALID_CHARACTER_SEQUENCE = "invalid_character_sequence"


@dataclass
class ValidationError:
    """Represents a rule-based validation error"""
    error_type: ValidationErrorType
    position: int
    length: int
    text_segment: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of rule-based validation"""
    text: str
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    overall_score: float
    processing_time: float


class KhmerUnicodeRanges:
    """Unicode ranges and character classifications for Khmer script"""
    
    # Main Khmer Unicode ranges
    KHMER_CONSONANTS = range(0x1780, 0x17A4)  # ក-ហ, អ
    KHMER_INDEPENDENT_VOWELS = range(0x17A5, 0x17B4)  # ឥ-ឳ
    KHMER_DEPENDENT_VOWELS = range(0x17B6, 0x17D4)  # ា-៓
    KHMER_DIACRITICS = range(0x17C6, 0x17D4)  # ំ-៓
    KHMER_SIGNS = range(0x17D4, 0x17DD)  # ។-ៜ
    KHMER_DIGITS = range(0x17E0, 0x17EA)  # ០-៩
    
    # Special characters
    COENG = 0x17D2  # ្ (subscript consonant marker)
    ROBAT = 0x17CC  # ំ (robat - inherent vowel killer)
    TRIISAP = 0x17C9  # ៉ (triisap - aspiration mark)
    MUUSIKATOAN = 0x17CA  # ៊ (muusikatoan - aspiration mark)
    
    @classmethod
    def is_khmer_consonant(cls, char: str) -> bool:
        """Check if character is a Khmer consonant"""
        return ord(char) in cls.KHMER_CONSONANTS
    
    @classmethod
    def is_khmer_vowel(cls, char: str) -> bool:
        """Check if character is a Khmer vowel (dependent or independent)"""
        code = ord(char)
        return (code in cls.KHMER_INDEPENDENT_VOWELS or 
                code in cls.KHMER_DEPENDENT_VOWELS)
    
    @classmethod
    def is_khmer_diacritic(cls, char: str) -> bool:
        """Check if character is a Khmer diacritic"""
        return ord(char) in cls.KHMER_DIACRITICS
    
    @classmethod
    def is_coeng(cls, char: str) -> bool:
        """Check if character is coeng (subscript marker)"""
        return ord(char) == cls.COENG
    
    @classmethod
    def is_combining_mark(cls, char: str) -> bool:
        """Check if character is a combining mark"""
        code = ord(char)
        return (code in cls.KHMER_DIACRITICS or 
                code == cls.COENG or
                code == cls.ROBAT)


class KhmerSyllableValidator:
    """Validates Khmer syllable structure according to linguistic rules"""
    
    def __init__(self):
        self.logger = logging.getLogger("khmer_syllable_validator")
        
        # Valid syllable patterns (simplified)
        # C = Consonant, V = Vowel, D = Diacritic, S = Subscript
        self.valid_patterns = [
            r'^[\u1780-\u17A3]$',  # Single consonant
            r'^[\u1780-\u17A3][\u17B6-\u17D3]$',  # Consonant + vowel
            r'^[\u1780-\u17A3][\u17C6-\u17D3]$',  # Consonant + diacritic
            r'^[\u1780-\u17A3][\u17D2][\u1780-\u17A3]$',  # Consonant + coeng + consonant
            r'^[\u1780-\u17A3][\u17B6-\u17D3][\u17C6-\u17D3]$',  # C + V + D
            r'^[\u1780-\u17A3][\u17D2][\u1780-\u17A3][\u17B6-\u17D3]$',  # C + coeng + C + V
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(pattern) for pattern in self.valid_patterns]
    
    def validate_syllable_structure(self, syllable: str) -> List[ValidationError]:
        """Validate syllable structure against Khmer linguistic rules"""
        errors = []
        
        if not syllable or not syllable.strip():
            return errors
        
        # Check if syllable matches any valid pattern
        is_valid = any(pattern.match(syllable) for pattern in self.compiled_patterns)
        
        if not is_valid:
            # Analyze specific issues
            issues = self._analyze_syllable_issues(syllable)
            errors.extend(issues)
        
        return errors
    
    def _analyze_syllable_issues(self, syllable: str) -> List[ValidationError]:
        """Analyze specific structural issues in syllable"""
        errors = []
        
        # Check for orphaned combining marks
        if len(syllable) == 1 and KhmerUnicodeRanges.is_combining_mark(syllable):
            errors.append(ValidationError(
                error_type=ValidationErrorType.ORPHANED_COMBINING_MARK,
                position=0,
                length=1,
                text_segment=syllable,
                description=f"Orphaned combining mark '{syllable}' without base character",
                severity="error",
                suggestion="Combine with appropriate base consonant"
            ))
        
        # Check coeng usage
        coeng_errors = self._validate_coeng_usage(syllable)
        errors.extend(coeng_errors)
        
        # Check vowel combinations
        vowel_errors = self._validate_vowel_combinations(syllable)
        errors.extend(vowel_errors)
        
        return errors
    
    def _validate_coeng_usage(self, syllable: str) -> List[ValidationError]:
        """Validate coeng (subscript) usage"""
        errors = []
        
        coeng_positions = [i for i, char in enumerate(syllable) 
                          if KhmerUnicodeRanges.is_coeng(char)]
        
        for pos in coeng_positions:
            # Coeng must be followed by a consonant
            if pos + 1 >= len(syllable):
                errors.append(ValidationError(
                    error_type=ValidationErrorType.INVALID_COENG_USAGE,
                    position=pos,
                    length=1,
                    text_segment=syllable[pos],
                    description="Coeng at end of syllable without following consonant",
                    severity="error",
                    suggestion="Add consonant after coeng or remove coeng"
                ))
            elif not KhmerUnicodeRanges.is_khmer_consonant(syllable[pos + 1]):
                errors.append(ValidationError(
                    error_type=ValidationErrorType.INVALID_COENG_USAGE,
                    position=pos,
                    length=2,
                    text_segment=syllable[pos:pos+2],
                    description="Coeng not followed by valid consonant",
                    severity="error",
                    suggestion="Replace with valid consonant"
                ))
            
            # Coeng must be preceded by a consonant
            if pos == 0:
                errors.append(ValidationError(
                    error_type=ValidationErrorType.INVALID_COENG_USAGE,
                    position=pos,
                    length=1,
                    text_segment=syllable[pos],
                    description="Coeng at beginning of syllable",
                    severity="error",
                    suggestion="Add base consonant before coeng"
                ))
            elif not KhmerUnicodeRanges.is_khmer_consonant(syllable[pos - 1]):
                errors.append(ValidationError(
                    error_type=ValidationErrorType.INVALID_COENG_USAGE,
                    position=pos - 1,
                    length=2,
                    text_segment=syllable[pos-1:pos+1],
                    description="Coeng not preceded by valid consonant",
                    severity="error",
                    suggestion="Replace with valid consonant"
                ))
        
        return errors
    
    def _validate_vowel_combinations(self, syllable: str) -> List[ValidationError]:
        """Validate vowel combinations"""
        errors = []
        
        vowel_positions = [i for i, char in enumerate(syllable) 
                          if KhmerUnicodeRanges.is_khmer_vowel(char)]
        
        # Check for multiple dependent vowels (usually invalid)
        dependent_vowels = [i for i in vowel_positions 
                           if ord(syllable[i]) in KhmerUnicodeRanges.KHMER_DEPENDENT_VOWELS]
        
        if len(dependent_vowels) > 1:
            # Some combinations are valid (like េា), but most are not
            # This is a simplified check
            errors.append(ValidationError(
                error_type=ValidationErrorType.INVALID_VOWEL_COMBINATION,
                position=dependent_vowels[0],
                length=dependent_vowels[-1] - dependent_vowels[0] + 1,
                text_segment=syllable[dependent_vowels[0]:dependent_vowels[-1]+1],
                description="Multiple dependent vowels in syllable",
                severity="warning",
                suggestion="Check if vowel combination is valid"
            ))
        
        return errors


class KhmerSequenceValidator:
    """Validates Unicode character sequences in Khmer text"""
    
    def __init__(self):
        self.logger = logging.getLogger("khmer_sequence_validator")
    
    def validate_unicode_sequence(self, text: str) -> List[ValidationError]:
        """Validate Unicode character sequences"""
        errors = []
        
        for i, char in enumerate(text):
            # Check for invalid character combinations
            if i > 0:
                prev_char = text[i-1]
                current_errors = self._check_character_pair(prev_char, char, i-1)
                errors.extend(current_errors)
        
        return errors
    
    def _check_character_pair(self, prev_char: str, current_char: str, position: int) -> List[ValidationError]:
        """Check validity of character pair"""
        errors = []
        
        # Combining marks must follow appropriate base characters
        if KhmerUnicodeRanges.is_combining_mark(current_char):
            if not self._is_valid_base_for_combining(prev_char):
                errors.append(ValidationError(
                    error_type=ValidationErrorType.INVALID_CHARACTER_SEQUENCE,
                    position=position,
                    length=2,
                    text_segment=prev_char + current_char,
                    description=f"Combining mark '{current_char}' after invalid base '{prev_char}'",
                    severity="error",
                    suggestion="Use combining mark with valid base character"
                ))
        
        # Check for double coeng
        if (KhmerUnicodeRanges.is_coeng(prev_char) and 
            KhmerUnicodeRanges.is_coeng(current_char)):
            errors.append(ValidationError(
                error_type=ValidationErrorType.INVALID_UNICODE_SEQUENCE,
                position=position,
                length=2,
                text_segment=prev_char + current_char,
                description="Double coeng sequence",
                severity="error",
                suggestion="Remove one coeng"
            ))
        
        return errors
    
    def _is_valid_base_for_combining(self, char: str) -> bool:
        """Check if character can serve as base for combining marks"""
        return (KhmerUnicodeRanges.is_khmer_consonant(char) or 
                KhmerUnicodeRanges.is_khmer_vowel(char) or 
                char.isspace())


class RuleBasedValidator:
    """
    Comprehensive rule-based validator for Khmer text
    
    Combines syllable structure validation, Unicode sequence checking,
    and diacritic placement rules.
    """
    
    def __init__(self, 
                 validate_syllables: bool = True,
                 validate_sequences: bool = True,
                 strict_mode: bool = False):
        """
        Initialize rule-based validator
        
        Args:
            validate_syllables: Enable syllable structure validation
            validate_sequences: Enable Unicode sequence validation
            strict_mode: Use strict validation rules
        """
        self.validate_syllables = validate_syllables
        self.validate_sequences = validate_sequences
        self.strict_mode = strict_mode
        
        # Initialize sub-validators
        self.syllable_validator = KhmerSyllableValidator()
        self.sequence_validator = KhmerSequenceValidator()
        
        self.logger = logging.getLogger("rule_based_validator")
        self.logger.info(f"Initialized rule-based validator (strict={strict_mode})")
    
    def validate_text(self, text: str, syllables: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate text using rule-based approach
        
        Args:
            text: Text to validate
            syllables: Pre-segmented syllables (optional)
            
        Returns:
            Validation result with errors and overall score
        """
        import time
        start_time = time.time()
        
        all_errors = []
        
        # Syllable-level validation
        if self.validate_syllables and syllables:
            syllable_errors = self._validate_syllables(syllables)
            all_errors.extend(syllable_errors)
        
        # Sequence-level validation
        if self.validate_sequences:
            sequence_errors = self.sequence_validator.validate_unicode_sequence(text)
            all_errors.extend(sequence_errors)
        
        # Categorize errors
        errors = [e for e in all_errors if e.severity == "error"]
        warnings = [e for e in all_errors if e.severity == "warning"]
        
        # Calculate overall score
        total_chars = len(text)
        error_penalty = len(errors) * 2 + len(warnings)  # Errors count double
        overall_score = max(0.0, 1.0 - (error_penalty / max(total_chars, 1)))
        
        processing_time = time.time() - start_time
        
        return ValidationResult(
            text=text,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            overall_score=overall_score,
            processing_time=processing_time
        )
    
    def _validate_syllables(self, syllables: List[str]) -> List[ValidationError]:
        """Validate list of syllables"""
        all_errors = []
        current_position = 0
        
        for syllable in syllables:
            syllable_errors = self.syllable_validator.validate_syllable_structure(syllable)
            
            # Adjust positions to match original text
            for error in syllable_errors:
                error.position += current_position
            
            all_errors.extend(syllable_errors)
            current_position += len(syllable)
        
        return all_errors
    
    def get_validation_statistics(self, texts: List[str], syllables_list: Optional[List[List[str]]] = None) -> Dict:
        """Get validation statistics for multiple texts"""
        stats = {
            'total_texts': len(texts),
            'valid_texts': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'error_types': {},
            'average_score': 0.0,
            'processing_time': 0.0
        }
        
        total_score = 0.0
        
        for i, text in enumerate(texts):
            syllables = syllables_list[i] if syllables_list else None
            result = self.validate_text(text, syllables)
            
            if result.is_valid:
                stats['valid_texts'] += 1
            
            stats['total_errors'] += len(result.errors)
            stats['total_warnings'] += len(result.warnings)
            stats['processing_time'] += result.processing_time
            total_score += result.overall_score
            
            # Count error types
            for error in result.errors + result.warnings:
                error_type = error.error_type.value
                stats['error_types'][error_type] = stats['error_types'].get(error_type, 0) + 1
        
        stats['average_score'] = total_score / len(texts) if texts else 0.0
        stats['validation_rate'] = stats['valid_texts'] / len(texts) if texts else 0.0
        
        return stats
    
    def generate_report(self, validation_results: List[ValidationResult]) -> str:
        """Generate comprehensive validation report"""
        if not validation_results:
            return "No validation results to report"
        
        report = []
        report.append("📋 RULE-BASED VALIDATION REPORT")
        report.append("=" * 40)
        report.append("")
        
        # Overall statistics
        total_texts = len(validation_results)
        valid_texts = sum(1 for r in validation_results if r.is_valid)
        total_errors = sum(len(r.errors) for r in validation_results)
        total_warnings = sum(len(r.warnings) for r in validation_results)
        avg_score = sum(r.overall_score for r in validation_results) / total_texts
        
        report.append("📊 OVERALL STATISTICS")
        report.append("-" * 25)
        report.append(f"Total texts validated: {total_texts:,}")
        report.append(f"Valid texts: {valid_texts:,} ({valid_texts/total_texts:.1%})")
        report.append(f"Total errors: {total_errors:,}")
        report.append(f"Total warnings: {total_warnings:,}")
        report.append(f"Average score: {avg_score:.3f}")
        report.append("")
        
        # Error type breakdown
        error_types = {}
        for result in validation_results:
            for error in result.errors + result.warnings:
                error_type = error.error_type.value
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if error_types:
            report.append("🔍 ERROR TYPE BREAKDOWN")
            report.append("-" * 25)
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / (total_errors + total_warnings)) * 100
                report.append(f"{error_type}: {count:,} ({percentage:.1f}%)")
            report.append("")
        
        # Sample errors
        sample_errors = []
        for result in validation_results[:5]:  # First 5 results
            sample_errors.extend(result.errors[:3])  # Up to 3 errors each
        
        if sample_errors:
            report.append("📝 SAMPLE ERRORS")
            report.append("-" * 15)
            for i, error in enumerate(sample_errors[:10], 1):
                report.append(f"{i}. {error.description}")
                report.append(f"   Text: '{error.text_segment}' at position {error.position}")
                if error.suggestion:
                    report.append(f"   Suggestion: {error.suggestion}")
            report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Demo usage
    print("📋 RULE-BASED VALIDATOR DEMO")
    print("=" * 35)
    
    # Sample texts with various issues
    test_texts = [
        "នេះជាការសាកល្បងត្រឹមត្រូវ។",  # Correct text
        "កំ្",  # Invalid coeng usage
        "្ក",  # Coeng at beginning
        "កា៉ើ",  # Multiple diacritics
        "។",  # Standalone punctuation
    ]
    
    validator = RuleBasedValidator()
    
    print("Testing rule-based validation...")
    for i, text in enumerate(test_texts, 1):
        result = validator.validate_text(text)
        print(f"\nTest {i}: '{text}'")
        print(f"Valid: {result.is_valid}")
        print(f"Score: {result.overall_score:.3f}")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")
        
        if result.errors:
            print("Error details:")
            for error in result.errors:
                print(f"  - {error.description}")
    
    print("\nRule-based validation demo completed!") 