"""
Text Preprocessing Pipeline

This module provides the main preprocessing pipeline that combines
text cleaning, syllable segmentation, quality validation, and
statistical analysis for Khmer text corpus preparation.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Generator, Tuple
from dataclasses import dataclass
from datetime import datetime
import pickle
import json

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.text_processor import TextProcessor
from data_collection.file_loader import FileLoader, TextDocument
from word_cluster.subword_cluster import (
    khmer_syllables_no_regex_fast,
    khmer_syllables_no_regex,
    khmer_syllables,
    khmer_syllables_advanced
)


@dataclass
class ProcessingResult:
    """Result of text processing operation"""
    original_text: str
    cleaned_text: str
    syllables: List[str]
    word_count: int
    syllable_count: int
    character_count: int
    khmer_ratio: float
    quality_score: float
    processing_time: float
    is_valid: bool = True
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class TextPreprocessingPipeline:
    """Advanced text preprocessing pipeline for Khmer spellchecker development"""
    
    def __init__(
        self,
        segmentation_method: str = 'regex_basic',
        min_khmer_ratio: float = 0.7,
        min_text_length: int = 20,
        max_english_ratio: float = 0.2,
        quality_threshold: float = 0.6,
        batch_size: int = 1000
    ):
        self.segmentation_method = segmentation_method
        self.min_khmer_ratio = min_khmer_ratio
        self.min_text_length = min_text_length
        self.max_english_ratio = max_english_ratio
        self.quality_threshold = quality_threshold
        self.batch_size = batch_size
        
        # Initialize components
        self.text_processor = TextProcessor(
            min_khmer_ratio=min_khmer_ratio,
            min_length=min_text_length,
            max_english_ratio=max_english_ratio
        )
        
        # Select segmentation function
        self.segmentation_functions = {
            'no_regex_fast': khmer_syllables_no_regex_fast,
            'no_regex': khmer_syllables_no_regex,
            'regex_basic': khmer_syllables,
            'regex_advanced': khmer_syllables_advanced
        }
        
        if segmentation_method not in self.segmentation_functions:
            raise ValueError(f"Unknown segmentation method: {segmentation_method}")
        
        self.segment_text = self.segmentation_functions[segmentation_method]
        
        self.logger = logging.getLogger("text_pipeline")
        
        # Processing statistics
        self.stats = {
            'texts_processed': 0,
            'texts_accepted': 0,
            'texts_rejected': 0,
            'total_syllables': 0,
            'total_characters': 0,
            'total_processing_time': 0.0,
            'segmentation_errors': 0
        }
        
        self.logger.info(f"TextPreprocessingPipeline initialized with {segmentation_method} segmentation")
    
    def process_text(self, text: str) -> ProcessingResult:
        """Process a single text through the complete pipeline"""
        import time
        start_time = time.time()
        
        try:
            # Clean text
            cleaned_text = self.text_processor.clean_text(text)
            
            # Calculate quality metrics
            metrics = self.text_processor.calculate_quality_metrics(cleaned_text)
            
            # Validate quality
            is_valid = (
                metrics.length >= self.min_text_length and
                metrics.khmer_ratio >= self.min_khmer_ratio and
                metrics.quality_score >= self.quality_threshold
            )
            
            # Segment into syllables if valid
            syllables = []
            if is_valid and cleaned_text.strip():
                try:
                    syllables = self.segment_text(cleaned_text)
                    if not isinstance(syllables, list):
                        syllables = []
                        is_valid = False
                except Exception as e:
                    self.logger.warning(f"Segmentation error: {e}")
                    syllables = []
                    is_valid = False
                    self.stats['segmentation_errors'] += 1
            
            # Calculate additional metrics
            word_count = len(cleaned_text.split()) if cleaned_text else 0
            syllable_count = len(syllables)
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['texts_processed'] += 1
            if is_valid:
                self.stats['texts_accepted'] += 1
                self.stats['total_syllables'] += syllable_count
                self.stats['total_characters'] += metrics.length
            else:
                self.stats['texts_rejected'] += 1
            
            self.stats['total_processing_time'] += processing_time
            
            return ProcessingResult(
                original_text=text,
                cleaned_text=cleaned_text,
                syllables=syllables,
                word_count=word_count,
                syllable_count=syllable_count,
                character_count=metrics.length,
                khmer_ratio=metrics.khmer_ratio,
                quality_score=metrics.quality_score,
                processing_time=processing_time,
                is_valid=is_valid
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Processing error: {e}")
            self.stats['texts_processed'] += 1
            self.stats['texts_rejected'] += 1
            self.stats['total_processing_time'] += processing_time
            
            return ProcessingResult(
                original_text=text,
                cleaned_text="",
                syllables=[],
                word_count=0,
                syllable_count=0,
                character_count=0,
                khmer_ratio=0.0,
                quality_score=0.0,
                processing_time=processing_time,
                is_valid=False,
                errors=[str(e)]
            )
    
    def process_texts_batch(self, texts: List[str]) -> List[ProcessingResult]:
        """Process a batch of texts"""
        results = []
        
        for i, text in enumerate(texts):
            result = self.process_text(text)
            results.append(result)
            
            # Progress logging
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1}/{len(texts)} texts in batch")
        
        return results
    
    def process_document(self, document: TextDocument) -> List[ProcessingResult]:
        """Process a complete document by splitting into sentences/paragraphs"""
        from data_collection.file_loader import FileLoader
        
        loader = FileLoader("")  # Just for utility functions
        
        # Split document into paragraphs
        paragraphs = loader.split_into_paragraphs(document, min_length=self.min_text_length)
        
        self.logger.info(f"Processing document {document.filename}: {len(paragraphs)} paragraphs")
        
        # Process paragraphs in batches
        results = []
        for i in range(0, len(paragraphs), self.batch_size):
            batch = paragraphs[i:i + self.batch_size]
            batch_results = self.process_texts_batch(batch)
            results.extend(batch_results)
            
            self.logger.info(f"Completed batch {i//self.batch_size + 1}/{(len(paragraphs)-1)//self.batch_size + 1}")
        
        return results
    
    def extract_valid_texts(self, results: List[ProcessingResult]) -> List[str]:
        """Extract only valid cleaned texts from processing results"""
        return [result.cleaned_text for result in results if result.is_valid and result.cleaned_text.strip()]
    
    def extract_syllables(self, results: List[ProcessingResult]) -> List[List[str]]:
        """Extract syllable sequences from processing results"""
        return [result.syllables for result in results if result.is_valid and result.syllables]
    
    def get_processing_summary(self, results: List[ProcessingResult]) -> Dict:
        """Generate summary statistics for processed results"""
        if not results:
            return {}
        
        valid_results = [r for r in results if r.is_valid]
        
        total_texts = len(results)
        valid_texts = len(valid_results)
        
        if valid_results:
            avg_quality = sum(r.quality_score for r in valid_results) / valid_texts
            avg_khmer_ratio = sum(r.khmer_ratio for r in valid_results) / valid_texts
            avg_syllables = sum(r.syllable_count for r in valid_results) / valid_texts
            total_syllables = sum(r.syllable_count for r in valid_results)
            total_characters = sum(r.character_count for r in valid_results)
        else:
            avg_quality = avg_khmer_ratio = avg_syllables = 0
            total_syllables = total_characters = 0
        
        return {
            'total_texts': total_texts,
            'valid_texts': valid_texts,
            'rejected_texts': total_texts - valid_texts,
            'acceptance_rate': valid_texts / total_texts if total_texts > 0 else 0,
            'avg_quality_score': avg_quality,
            'avg_khmer_ratio': avg_khmer_ratio,
            'avg_syllables_per_text': avg_syllables,
            'total_syllables': total_syllables,
            'total_characters': total_characters,
            'segmentation_method': self.segmentation_method
        }
    
    def save_results(self, results: List[ProcessingResult], output_path: str):
        """Save processing results to file"""
        # Save as pickle for complete data
        pickle_path = f"{output_path}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary as JSON
        summary = self.get_processing_summary(results)
        json_path = f"{output_path}_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save valid texts
        valid_texts = self.extract_valid_texts(results)
        text_path = f"{output_path}_clean.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            for text in valid_texts:
                f.write(text + '\n\n')
        
        self.logger.info(f"Results saved to {pickle_path}, {json_path}, and {text_path}")
    
    def _log_statistics(self):
        """Log processing statistics"""
        self.logger.info("=== Processing Statistics ===")
        for key, value in self.stats.items():
            if isinstance(value, float):
                self.logger.info(f"{key}: {value:.3f}")
            else:
                self.logger.info(f"{key}: {value:,}")


class CorpusProcessor:
    """High-level processor for entire text corpus"""
    
    def __init__(
        self,
        data_directory: str,
        output_directory: str = "output/preprocessing",
        **pipeline_kwargs
    ):
        self.data_directory = data_directory
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.file_loader = FileLoader(data_directory, max_file_size=500*1024*1024)
        self.pipeline = TextPreprocessingPipeline(**pipeline_kwargs)
        
        self.logger = logging.getLogger("corpus_processor")
        
        self.logger.info(f"CorpusProcessor initialized for {data_directory}")
    
    def process_corpus(self, max_files: Optional[int] = None, save_intermediate: bool = True) -> Dict:
        """Process entire corpus and return summary statistics"""
        self.logger.info("Starting corpus processing...")
        
        # Load documents
        documents = self.file_loader.load_all_files(max_files=max_files)
        
        if not documents:
            self.logger.error("No documents loaded")
            return {}
        
        all_results = []
        file_summaries = []
        
        # Process each document
        for i, document in enumerate(documents, 1):
            self.logger.info(f"Processing document {i}/{len(documents)}: {document.filename}")
            
            # Process document
            results = self.pipeline.process_document(document)
            all_results.extend(results)
            
            # Generate file summary
            file_summary = self.pipeline.get_processing_summary(results)
            file_summary['filename'] = document.filename
            file_summary['original_size_mb'] = document.size / (1024 * 1024)
            file_summaries.append(file_summary)
            
            # Save intermediate results if requested
            if save_intermediate:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.output_directory / f"processed_{document.filename}_{timestamp}"
                self.pipeline.save_results(results, str(output_path))
            
            self.logger.info(f"Completed {document.filename}: {file_summary['valid_texts']}/{file_summary['total_texts']} texts accepted")
        
        # Generate overall corpus summary
        corpus_summary = self.pipeline.get_processing_summary(all_results)
        corpus_summary['file_summaries'] = file_summaries
        corpus_summary['total_files'] = len(documents)
        corpus_summary['processing_timestamp'] = datetime.now().isoformat()
        
        # Save final results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_output_path = self.output_directory / f"corpus_processed_{timestamp}"
        self.pipeline.save_results(all_results, str(final_output_path))
        
        # Save corpus summary
        summary_path = self.output_directory / f"corpus_summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info("Corpus processing completed!")
        self.logger.info(f"Summary: {corpus_summary['valid_texts']:,} valid texts from {corpus_summary['total_texts']:,} total")
        
        return corpus_summary


def quick_preview(data_directory: str, num_samples: int = 5):
    """Quick preview of corpus preprocessing"""
    from data_collection.file_loader import get_corpus_preview
    
    print("🔍 Quick Corpus Preview")
    print("=" * 50)
    
    # Show file overview
    get_corpus_preview(data_directory, num_files=1, preview_lines=5)
    
    # Test preprocessing pipeline
    print(f"\n🔧 Testing Preprocessing Pipeline")
    print("-" * 30)
    
    pipeline = TextPreprocessingPipeline()
    loader = FileLoader(data_directory)
    
    # Load small sample
    files = loader.get_file_list()
    if files:
        # Read first few lines for testing
        with open(files[0], 'r', encoding='utf-8', errors='replace') as f:
            lines = [f.readline().strip() for _ in range(num_samples)]
        
        for i, line in enumerate(lines, 1):
            if line:
                print(f"\n📝 Sample {i}:")
                result = pipeline.process_text(line)
                print(f"   Valid: {result.is_valid}")
                print(f"   Khmer ratio: {result.khmer_ratio:.2f}")
                print(f"   Quality score: {result.quality_score:.2f}")
                print(f"   Syllables: {result.syllable_count}")
                if result.syllables[:5]:  # Show first 5 syllables
                    print(f"   First syllables: {' | '.join(result.syllables[:5])}")
    
    print("\n✅ Preview completed!")


if __name__ == "__main__":
    # Example usage
    data_dir = r"C:\temps\ML-data"
    
    # Quick preview
    quick_preview(data_dir)
    
    # Process corpus (uncomment to run)
    # processor = CorpusProcessor(data_dir, max_files=1)
    # summary = processor.process_corpus()
    # print(f"Processed {summary['valid_texts']} texts") 