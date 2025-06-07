"""
File Loader Module

This module provides tools for loading and processing large text files
from local directories for Khmer text corpus building.
"""

import os
import logging
from typing import List, Dict, Generator, Optional, Tuple
from pathlib import Path
import chardet
from dataclasses import dataclass
import hashlib


@dataclass
class TextDocument:
    """Data class for storing document information"""
    content: str
    filename: str
    filepath: str
    size: int
    encoding: str = "utf-8"
    line_count: int = 0
    char_count: int = 0
    content_hash: str = ""
    
    def __post_init__(self):
        """Generate metrics after initialization"""
        if self.content:
            self.char_count = len(self.content)
            self.line_count = self.content.count('\n') + 1
            if not self.content_hash:
                self.content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()


class FileLoader:
    """Advanced file loading system for large text corpora"""
    
    def __init__(
        self,
        data_directory: str,
        supported_extensions: List[str] = None,
        chunk_size: int = 1024 * 1024,  # 1MB chunks
        encoding_detection: bool = True,
        max_file_size: int = 500 * 1024 * 1024  # 500MB max per file
    ):
        self.data_directory = Path(data_directory)
        self.supported_extensions = supported_extensions or ['.txt', '.text']
        self.chunk_size = chunk_size
        self.encoding_detection = encoding_detection
        self.max_file_size = max_file_size
        
        self.logger = logging.getLogger("file_loader")
        
        # Statistics
        self.stats = {
            'files_found': 0,
            'files_loaded': 0,
            'files_failed': 0,
            'total_size_processed': 0,
            'total_lines_processed': 0,
            'encoding_errors': 0
        }
        
        # Validate directory
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {data_directory}")
        
        self.logger.info(f"FileLoader initialized for directory: {self.data_directory}")
    
    def detect_encoding(self, filepath: Path) -> str:
        """Detect file encoding using chardet"""
        try:
            # Read a sample of the file for encoding detection
            sample_size = min(32768, filepath.stat().st_size)  # 32KB sample
            
            with open(filepath, 'rb') as f:
                raw_data = f.read(sample_size)
            
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0.0)
            
            self.logger.debug(f"Detected encoding for {filepath.name}: {encoding} (confidence: {confidence:.2f})")
            
            # Fallback to utf-8 if confidence is too low
            if confidence < 0.7:
                self.logger.warning(f"Low confidence ({confidence:.2f}) for {filepath.name}, using utf-8")
                encoding = 'utf-8'
            
            return encoding
            
        except Exception as e:
            self.logger.warning(f"Encoding detection failed for {filepath.name}: {e}")
            return 'utf-8'
    
    def get_file_list(self) -> List[Path]:
        """Get list of supported text files in directory"""
        files = []
        
        for ext in self.supported_extensions:
            pattern = f"*{ext}"
            found_files = list(self.data_directory.glob(pattern))
            files.extend(found_files)
        
        # Sort by file size (process smaller files first for testing)
        files.sort(key=lambda f: f.stat().st_size)
        
        self.stats['files_found'] = len(files)
        self.logger.info(f"Found {len(files)} text files")
        
        return files
    
    def load_file_content(self, filepath: Path) -> Optional[TextDocument]:
        """Load content from a single file"""
        try:
            file_size = filepath.stat().st_size
            
            # Check file size limit
            if file_size > self.max_file_size:
                self.logger.warning(f"File {filepath.name} too large ({file_size/1024/1024:.1f}MB), skipping")
                return None
            
            # Detect encoding if requested
            if self.encoding_detection:
                encoding = self.detect_encoding(filepath)
            else:
                encoding = 'utf-8'
            
            # Read file content
            self.logger.info(f"Loading {filepath.name} ({file_size/1024/1024:.1f}MB)")
            
            with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Create document object
            document = TextDocument(
                content=content,
                filename=filepath.name,
                filepath=str(filepath),
                size=file_size,
                encoding=encoding
            )
            
            self.stats['files_loaded'] += 1
            self.stats['total_size_processed'] += file_size
            self.stats['total_lines_processed'] += document.line_count
            
            self.logger.info(f"Successfully loaded {filepath.name}: {document.line_count:,} lines, {document.char_count:,} characters")
            
            return document
            
        except UnicodeDecodeError as e:
            self.stats['encoding_errors'] += 1
            self.stats['files_failed'] += 1
            self.logger.error(f"Encoding error in {filepath.name}: {e}")
            return None
            
        except Exception as e:
            self.stats['files_failed'] += 1
            self.logger.error(f"Error loading {filepath.name}: {e}")
            return None
    
    def load_file_chunks(self, filepath: Path) -> Generator[str, None, None]:
        """Load file content in chunks for memory-efficient processing"""
        try:
            # Detect encoding
            encoding = self.detect_encoding(filepath) if self.encoding_detection else 'utf-8'
            
            self.logger.info(f"Loading {filepath.name} in chunks of {self.chunk_size/1024:.0f}KB")
            
            with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
                        
        except Exception as e:
            self.logger.error(f"Error reading chunks from {filepath.name}: {e}")
            return
    
    def load_all_files(self, max_files: Optional[int] = None) -> List[TextDocument]:
        """Load all text files from directory"""
        files = self.get_file_list()
        
        if max_files:
            files = files[:max_files]
            self.logger.info(f"Processing first {len(files)} files (limited by max_files)")
        
        documents = []
        
        for i, filepath in enumerate(files, 1):
            self.logger.info(f"Processing file {i}/{len(files)}: {filepath.name}")
            
            document = self.load_file_content(filepath)
            if document:
                documents.append(document)
            
            # Progress update
            if i % 5 == 0 or i == len(files):
                self.logger.info(f"Progress: {i}/{len(files)} files processed")
        
        self.logger.info(f"File loading completed: {len(documents)} documents loaded")
        self._log_statistics()
        
        return documents
    
    def split_into_paragraphs(self, document: TextDocument, min_length: int = 50) -> List[str]:
        """Split document content into paragraphs"""
        # Split by double newlines (standard paragraph separator)
        paragraphs = document.content.split('\n\n')
        
        # Filter out short paragraphs and clean whitespace
        clean_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) >= min_length:
                clean_paragraphs.append(para)
        
        self.logger.debug(f"Split {document.filename} into {len(clean_paragraphs)} paragraphs")
        return clean_paragraphs
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (basic approach for Khmer)"""
        import re
        
        # Split on common sentence endings
        sentences = re.split(r'[.!?។]+', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def get_corpus_statistics(self, documents: List[TextDocument]) -> Dict:
        """Calculate corpus-wide statistics"""
        if not documents:
            return {}
        
        total_chars = sum(doc.char_count for doc in documents)
        total_lines = sum(doc.line_count for doc in documents)
        total_files = len(documents)
        total_size = sum(doc.size for doc in documents)
        
        # Calculate averages
        avg_chars_per_file = total_chars / total_files
        avg_lines_per_file = total_lines / total_files
        avg_chars_per_line = total_chars / total_lines if total_lines > 0 else 0
        
        return {
            'total_files': total_files,
            'total_characters': total_chars,
            'total_lines': total_lines,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'avg_characters_per_file': avg_chars_per_file,
            'avg_lines_per_file': avg_lines_per_file,
            'avg_characters_per_line': avg_chars_per_line,
            'largest_file': max(documents, key=lambda d: d.size).filename,
            'smallest_file': min(documents, key=lambda d: d.size).filename
        }
    
    def _log_statistics(self):
        """Log loading statistics"""
        self.logger.info("=== File Loading Statistics ===")
        for key, value in self.stats.items():
            self.logger.info(f"{key}: {value:,}")
        
        if self.stats['total_size_processed'] > 0:
            size_mb = self.stats['total_size_processed'] / (1024 * 1024)
            self.logger.info(f"total_size_mb: {size_mb:.1f}")


def load_khmer_corpus(data_directory: str, **kwargs) -> List[TextDocument]:
    """Convenience function to load Khmer text corpus"""
    loader = FileLoader(data_directory, **kwargs)
    return loader.load_all_files()


def get_corpus_preview(data_directory: str, num_files: int = 1, preview_lines: int = 10):
    """Get a preview of the corpus content"""
    loader = FileLoader(data_directory)
    files = loader.get_file_list()
    
    print(f"📁 Corpus Preview: {data_directory}")
    print(f"📊 Found {len(files)} text files")
    print("=" * 60)
    
    for i, filepath in enumerate(files[:num_files]):
        print(f"\n🗂️  File {i+1}: {filepath.name}")
        print(f"📏 Size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")
        
        try:
            # Read first few lines
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                lines = []
                for _ in range(preview_lines):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line.strip())
            
            print("📄 Content preview:")
            for j, line in enumerate(lines, 1):
                if line:  # Only show non-empty lines
                    preview = line[:100] + "..." if len(line) > 100 else line
                    print(f"   {j:2d}: {preview}")
                    
        except Exception as e:
            print(f"   ❌ Error reading file: {e}") 