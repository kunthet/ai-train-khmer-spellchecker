#!/usr/bin/env python3
"""
Demo Script for Phase 1.3: Text Preprocessing Pipeline

This script demonstrates the complete preprocessing pipeline using
local text files instead of web scraping, including syllable segmentation,
statistical analysis, and quality validation.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from data_collection.file_loader import FileLoader, get_corpus_preview
from preprocessing.text_pipeline import (
    TextPreprocessingPipeline, 
    CorpusProcessor,
    quick_preview
)
from preprocessing.statistical_analyzer import StatisticalAnalyzer


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup file handler
    log_filename = f"logs/preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def demo_file_loading(data_directory: str):
    """Demonstrate file loading capabilities"""
    print("📁 DEMO: File Loading from Local Directory")
    print("=" * 50)
    
    # Show corpus preview
    get_corpus_preview(data_directory, num_files=1, preview_lines=3)
    
    # Load files with statistics
    loader = FileLoader(data_directory)
    files = loader.get_file_list()
    
    print(f"\n📊 File Analysis:")
    total_size = 0
    for i, filepath in enumerate(files, 1):
        size_mb = filepath.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"   {i}. {filepath.name}: {size_mb:.1f} MB")
    
    print(f"\n📈 Total corpus size: {total_size:.1f} MB")
    print(f"📈 Number of files: {len(files)}")
    
    return files


def demo_text_processing(data_directory: str, num_samples: int = 10):
    """Demonstrate text processing pipeline"""
    print("\n🔧 DEMO: Text Processing Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = TextPreprocessingPipeline(
        segmentation_method='no_regex_fast',
        min_khmer_ratio=0.6,
        min_text_length=20
    )
    
    # Load a small sample
    loader = FileLoader(data_directory)
    files = loader.get_file_list()
    
    if not files:
        print("❌ No files found!")
        return
    
    # Read sample lines from first file
    print(f"📝 Processing samples from: {files[0].name}")
    
    with open(files[0], 'r', encoding='utf-8', errors='replace') as f:
        lines = []
        for _ in range(num_samples * 2):  # Read extra to filter out short lines
            line = f.readline().strip()
            if line and len(line) > 30:  # Only meaningful lines
                lines.append(line)
            if len(lines) >= num_samples:
                break
    
    print(f"\n📊 Processing {len(lines)} sample texts...")
    
    valid_count = 0
    total_syllables = 0
    
    for i, line in enumerate(lines, 1):
        result = pipeline.process_text(line)
        
        if result.is_valid:
            valid_count += 1
            total_syllables += result.syllable_count
        
        print(f"\n📝 Sample {i}:")
        print(f"   Text: {line[:80]}{'...' if len(line) > 80 else ''}")
        print(f"   ✅ Valid: {result.is_valid}")
        print(f"   📊 Khmer ratio: {result.khmer_ratio:.3f}")
        print(f"   ⭐ Quality score: {result.quality_score:.3f}")
        print(f"   🔤 Syllables: {result.syllable_count}")
        
        if result.syllables and len(result.syllables) >= 5:
            sample_syllables = ' | '.join(result.syllables[:5])
            print(f"   🔣 First syllables: {sample_syllables}")
    
    print(f"\n📈 Processing Summary:")
    print(f"   Valid texts: {valid_count}/{len(lines)} ({valid_count/len(lines)*100:.1f}%)")
    print(f"   Total syllables: {total_syllables:,}")
    print(f"   Average syllables per valid text: {total_syllables/valid_count if valid_count > 0 else 0:.1f}")


def demo_statistical_analysis(data_directory: str, max_texts: int = 100):
    """Demonstrate statistical analysis"""
    print("\n📊 DEMO: Statistical Analysis")
    print("=" * 50)
    
    # Load sample data
    loader = FileLoader(data_directory)
    files = loader.get_file_list()
    
    if not files:
        print("❌ No files found!")
        return
    
    print(f"📖 Loading sample texts from: {files[0].name}")
    
    # Read sample paragraphs
    with open(files[0], 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Split into paragraphs and take sample
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
    sample_texts = paragraphs[:max_texts]
    
    print(f"📊 Analyzing {len(sample_texts)} text samples...")
    
    # Process texts to get syllables
    pipeline = TextPreprocessingPipeline()
    syllable_sequences = []
    valid_texts = []
    
    for text in sample_texts:
        result = pipeline.process_text(text)
        if result.is_valid:
            valid_texts.append(result.cleaned_text)
            syllable_sequences.append(result.syllables)
    
    print(f"✅ {len(valid_texts)} texts passed quality validation")
    
    if not valid_texts:
        print("❌ No valid texts for analysis!")
        return
    
    # Perform statistical analysis
    analyzer = StatisticalAnalyzer()
    analyzer.analyze_corpus(valid_texts, syllable_sequences)
    
    total_size = sum(len(text.encode('utf-8')) for text in valid_texts)
    stats = analyzer.get_corpus_statistics(len(valid_texts), total_size)
    
    # Generate and display report
    report = analyzer.generate_report(stats)
    print("\n" + report)
    
    return stats


def demo_corpus_processing(data_directory: str, max_files: int = 1):
    """Demonstrate full corpus processing (limited for demo)"""
    print("\n🏭 DEMO: Full Corpus Processing")
    print("=" * 50)
    
    # Initialize corpus processor
    processor = CorpusProcessor(
        data_directory=data_directory,
        output_directory="output/demo_preprocessing",
        segmentation_method='regex_advanced',
        min_khmer_ratio=0.6
    )
    
    print(f"🔄 Processing corpus (limited to {max_files} files for demo)...")
    
    # Process corpus
    summary = processor.process_corpus(max_files=max_files, save_intermediate=False)
    
    if summary:
        print(f"\n✅ Corpus processing completed!")
        print(f"📊 Summary:")
        print(f"   Total texts processed: {summary['total_texts']:,}")
        print(f"   Valid texts: {summary['valid_texts']:,}")
        print(f"   Acceptance rate: {summary['acceptance_rate']:.3f}")
        print(f"   Total syllables: {summary['total_syllables']:,}")
        print(f"   Total characters: {summary['total_characters']:,}")
        
        # Show file outputs
        output_dir = Path("output/demo_preprocessing")
        if output_dir.exists():
            files = list(output_dir.glob("*"))
            print(f"\n📁 Generated files ({len(files)}):")
            for file in sorted(files)[:5]:  # Show first 5 files
                size_kb = file.stat().st_size / 1024
                print(f"   {file.name}: {size_kb:.1f} KB")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more files")
    
    return summary


def main():
    """Main demonstration function"""
    # Configuration
    data_directory = r"C:\temps\ML-data"
    
    # Setup logging
    logger = setup_logging(logging.INFO)
    logger.info("Starting Phase 1.3: Text Preprocessing Pipeline Demo")
    
    print("🚀 PHASE 1.3: TEXT PREPROCESSING PIPELINE DEMO")
    print("=" * 60)
    print(f"📂 Data directory: {data_directory}")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    try:
        # Demo 1: File Loading
        files = demo_file_loading(data_directory)
        
        if not files:
            print("❌ No files found in directory. Please check the path!")
            return
        
        # Demo 2: Text Processing
        demo_text_processing(data_directory, num_samples=5)
        
        # Demo 3: Statistical Analysis
        stats = demo_statistical_analysis(data_directory, max_texts=50)
        
        # Demo 4: Full Corpus Processing (limited)
        print("\n" + "="*60)
        print("⚠️  NOTE: Full corpus processing demo is limited to 1 file")
        print("   For production, remove max_files limit to process all files")
        print("="*60)
        
        summary = demo_corpus_processing(data_directory, max_files=1)
        
        print("\n🎉 Phase 1.3 Demo completed successfully!")
        print("\n📋 Next Steps:")
        print("   1. Review generated output files in output/demo_preprocessing/")
        print("   2. Adjust preprocessing parameters as needed")
        print("   3. Process full corpus by removing file limits")
        print("   4. Move to Phase 1.4: Syllable Segmentation Integration")
        
    except FileNotFoundError:
        print(f"❌ Error: Directory not found: {data_directory}")
        print("   Please verify the path to your text files.")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main() 