#!/usr/bin/env python3
"""
Demo Script for Phase 1.4: Syllable Segmentation Integration

This script demonstrates the enhanced syllable segmentation API, frequency analysis,
and validation systems using regex_advanced as the default method.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from word_cluster.syllable_api import (
    SyllableSegmentationAPI, 
    SegmentationMethod
)
from word_cluster.syllable_frequency_analyzer import SyllableFrequencyAnalyzer
from data_collection.file_loader import FileLoader


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup file handler
    log_filename = f"logs/phase14_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def demo_syllable_api():
    """Demonstrate the new syllable segmentation API"""
    print("🔧 DEMO: Enhanced Syllable Segmentation API")
    print("=" * 50)
    
    # Initialize API with regex_advanced as default
    api = SyllableSegmentationAPI(SegmentationMethod.REGEX_ADVANCED)
    
    # Test texts
    test_texts = [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
        "ខ្ញុំស្រលាញ់ភាសាខ្មែរណាស់។",
        "ប្រទេសកម្ពុជាមានទីរដ្ឋធានីភ្នំពេញ។",
        "Hello world មិនមែនជាភាសាខ្មែរ",
        "123 បីមួយបី"
    ]
    
    print("📝 Single Text Segmentation:")
    for i, text in enumerate(test_texts, 1):
        result = api.segment_text(text)
        print(f"\n{i}. Text: {text}")
        print(f"   Success: {result.success}")
        print(f"   Syllables ({result.syllable_count}): {' | '.join(result.syllables)}")
        print(f"   Processing time: {result.processing_time:.4f}s")
        
        if not result.success:
            print(f"   Error: {result.error_message}")
    
    # Batch processing demo
    print(f"\n📦 Batch Processing:")
    batch_result = api.segment_batch(test_texts, show_progress=False)
    print(f"Total texts: {batch_result.total_texts}")
    print(f"Successful: {batch_result.successful_texts}")
    print(f"Failed: {batch_result.failed_texts}")
    print(f"Total syllables: {batch_result.total_syllables}")
    print(f"Average processing time: {batch_result.avg_processing_time:.4f}s")
    
    # Method comparison
    print(f"\n🔍 Method Comparison:")
    comparison_text = "នេះជាការសាកល្បងអត្ថបទខ្មែរ។"
    comparison = api.compare_methods(comparison_text)
    
    for method, result in comparison.items():
        print(f"{method:15s}: {' | '.join(result.syllables)} ({result.processing_time:.4f}s)")
    
    # Consistency validation
    print(f"\n✅ Consistency Validation:")
    is_consistent, details = api.validate_consistency(comparison_text)
    print(f"All methods consistent: {is_consistent}")
    
    if not is_consistent:
        print("Differences found:")
        for method, is_same in details["comparison"].items():
            print(f"  {method}: {'✅' if is_same else '❌'}")
    
    # API statistics
    print(f"\n📊 API Statistics:")
    stats = api.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return api


def demo_frequency_analysis(data_directory: str):
    """Demonstrate syllable frequency analysis"""
    print("\n📊 DEMO: Syllable Frequency Analysis")
    print("=" * 50)
    
    # Initialize frequency analyzer with regex_advanced
    analyzer = SyllableFrequencyAnalyzer(SegmentationMethod.REGEX_ADVANCED)
    
    # Load sample data
    print("📖 Loading sample texts...")
    loader = FileLoader(data_directory)
    files = loader.get_file_list()
    
    if not files:
        print("❌ No files found! Using sample texts instead.")
        sample_texts = [
            "នេះជាការសាកល្បងអត្ថបទខ្មែរ។ ខ្ញុំកំពុងសាកល្បងប្រព័ន្ធដែលបានកែលម្អ។",
            "ប្រទេសកម្ពុជាមានទីរដ្ឋធានីភ្នំពេញ។ ជាទីក្រុងធំបំផុតនៅកម្ពុជា។",
            "ភាសាខ្មែរមានអក្សរក្រម៧៤អក្សរ។ ជាភាសាមួយដែលមានប្រវត្តិយូរយារ។",
            "វប្បធម៌ខ្មែរមានភាពចម្រុះ និងសម្បូរបែប។ មានការបុរាណនិងទំនើប។",
            "សាសនាព្រះពុទ្ធជាសាសនាចម្បងរបស់ប្រជាជនខ្មែរ។ មានឥទ្ធិពលយ៉ាងធំ។"
        ]
    else:
        # Read sample from first file
        print(f"📖 Loading from: {files[0].name}")
        with open(files[0], 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Split into paragraphs and take first 20
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        sample_texts = paragraphs[:20]
    
    print(f"📊 Training on {len(sample_texts)} texts...")
    
    # Train frequency analyzer
    stats = analyzer.train_on_texts(sample_texts, show_progress=False)
    
    # Display results
    print(f"\n📈 Training Results:")
    print(f"Total syllables: {stats.total_syllables:,}")
    print(f"Unique syllables: {stats.unique_syllables:,}")
    print(f"Average syllables per text: {stats.avg_syllables_per_text:.1f}")
    print(f"Entropy: {stats.entropy:.3f}")
    
    # Show most common syllables
    print(f"\n🔝 Top 10 Most Common Syllables:")
    for i, (syllable, frequency) in enumerate(stats.most_common_syllables[:10], 1):
        percentage = (frequency / stats.total_syllables) * 100
        print(f"  {i:2d}. '{syllable}': {frequency:,} ({percentage:.2f}%)")
    
    # Coverage statistics
    print(f"\n📈 Coverage Statistics:")
    for percentile, syllable_count in sorted(stats.coverage_percentiles.items()):
        print(f"  {percentile}% of usage covered by top {syllable_count:,} syllables")
    
    # Test validation
    print(f"\n🧪 Syllable Validation Tests:")
    test_syllables = ["នេះ", "ការ", "ជា", "បាន", "xxxxx", "អាប់"]
    
    for syllable in test_syllables:
        validation = analyzer.validate_syllable(syllable)
        print(f"  '{syllable}': {validation.rarity_level} (freq: {validation.frequency}, confidence: {validation.confidence_score:.3f})")
        
        if validation.suggestions:
            suggestions = ', '.join([f"'{s}'" for s, _ in validation.suggestions[:3]])
            print(f"    Suggestions: {suggestions}")
    
    # Text validation demo
    print(f"\n📝 Text Validation Demo:")
    test_text = "នេះជាកាសាកល្បងអត្ថបទខ្មែរដែលមានពាក្យមិនត្រឹមត្រូវ។"
    text_validation = analyzer.validate_text(test_text)
    
    if text_validation["success"]:
        print(f"Text: {test_text}")
        print(f"Total syllables: {text_validation['total_syllables']}")
        print(f"Valid syllables: {text_validation['valid_syllables']}")
        print(f"Validity rate: {text_validation['validity_rate']:.1%}")
        print(f"Unknown syllables: {text_validation['unknown_syllables']}")
    
    # Generate full report
    print(f"\n📋 Full Frequency Analysis Report:")
    print("-" * 50)
    report = analyzer.generate_report()
    print(report)
    
    return analyzer


def demo_integration_with_preprocessing(data_directory: str):
    """Demonstrate integration with existing preprocessing pipeline"""
    print("\n🔗 DEMO: Integration with Preprocessing Pipeline")
    print("=" * 50)
    
    try:
        from preprocessing.text_pipeline import TextPreprocessingPipeline
        
        # Create pipeline with regex_advanced (now default)
        pipeline = TextPreprocessingPipeline()
        print(f"✅ Created preprocessing pipeline with default method")
        
        # Load sample data
        loader = FileLoader(data_directory)
        files = loader.get_file_list()
        
        if files:
            # Read small sample
            with open(files[0], 'r', encoding='utf-8', errors='replace') as f:
                lines = []
                for _ in range(10):
                    line = f.readline().strip()
                    if line and len(line) > 30:
                        lines.append(line)
                    if len(lines) >= 5:
                        break
            
            print(f"📊 Processing {len(lines)} sample texts:")
            
            for i, line in enumerate(lines, 1):
                result = pipeline.process_text(line)
                print(f"\n{i}. Text: {line[:60]}{'...' if len(line) > 60 else ''}")
                print(f"   Valid: {result.is_valid}")
                print(f"   Syllables: {result.syllable_count}")
                print(f"   Quality: {result.quality_score:.3f}")
                print(f"   Method: regex_advanced")
        
        else:
            print("❌ No files found for integration testing")
            
    except ImportError:
        print("❌ Preprocessing module not available for integration testing")


def demo_performance_comparison():
    """Compare performance of different segmentation methods"""
    print("\n⚡ DEMO: Performance Comparison")
    print("=" * 50)
    
    # Test text (longer for performance testing)
    test_text = "នេះជាការសាកល្បងអត្ថបទខ្មែរដ៏វែងមួយដែលមានមាតិកាច្រើនក្នុងគោលបំណងសាកល្បងលេខឲ្យក្រុមការងារយើងអាចវាស់វែងកার្យសម្រេចនៃប្រព័ន្ធដែលយើងបានអភិវឌ្ឍន៍ឡើង។ វាមានប្រយោជន៍ច្រើនសម្រាប់ការវិភាគនិងការប្រៀបធៀបប្រសិទ្ធភាពរបស់វិធីសាស្រ្តផ្សេងៗនៃការចែកតាងអក្ខរៈខ្មែរ។"
    
    api = SyllableSegmentationAPI()
    
    # Test all methods
    methods = [
        SegmentationMethod.NO_REGEX_FAST,
        SegmentationMethod.NO_REGEX,
        SegmentationMethod.REGEX_BASIC,
        SegmentationMethod.REGEX_ADVANCED
    ]
    
    print("🏃 Single text performance:")
    results = {}
    
    for method in methods:
        # Run multiple times for accurate timing
        times = []
        for _ in range(10):
            result = api.segment_text(test_text, method)
            times.append(result.processing_time)
        
        avg_time = sum(times) / len(times)
        results[method.value] = avg_time
        
        print(f"  {method.value:15s}: {avg_time:.4f}s avg, {result.syllable_count} syllables")
    
    # Batch performance test
    print(f"\n📦 Batch performance (100 texts):")
    batch_texts = [test_text] * 100
    
    for method in methods:
        batch_result = api.segment_batch(batch_texts, method, show_progress=False)
        throughput = batch_result.total_syllables / batch_result.total_processing_time
        
        print(f"  {method.value:15s}: {batch_result.total_processing_time:.2f}s total, {throughput:.0f} syllables/sec")
    
    # Recommend best method
    fastest_method = min(results.items(), key=lambda x: x[1])
    print(f"\n🏆 Fastest method: {fastest_method[0]} ({fastest_method[1]:.4f}s)")
    print(f"🎯 Current default: regex_advanced (recommended for accuracy)")


def main():
    """Main demonstration function"""
    # Configuration
    data_directory = r"C:\temps\ML-data"
    
    # Setup logging
    logger = setup_logging(logging.INFO)
    logger.info("Starting Phase 1.4: Syllable Segmentation Integration Demo")
    
    print("🚀 PHASE 1.4: SYLLABLE SEGMENTATION INTEGRATION DEMO")
    print("=" * 60)
    print(f"📂 Data directory: {data_directory}")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Default method: regex_advanced")
    print("")
    
    try:
        # Demo 1: Enhanced Syllable API
        api = demo_syllable_api()
        
        # Demo 2: Frequency Analysis
        analyzer = demo_frequency_analysis(data_directory)
        
        # Demo 3: Integration with existing preprocessing
        demo_integration_with_preprocessing(data_directory)
        
        # Demo 4: Performance comparison
        demo_performance_comparison()
        
        print("\n🎉 Phase 1.4 Demo completed successfully!")
        
        print("\n📋 Phase 1.4 Achievements:")
        print("   ✅ Enhanced syllable segmentation API with standardized interface")
        print("   ✅ Comprehensive error handling and performance monitoring")
        print("   ✅ Syllable frequency analysis and validation system")
        print("   ✅ Method comparison and consistency validation")
        print("   ✅ Integration with existing preprocessing pipeline")
        print("   ✅ regex_advanced set as default method")
        
        print("\n📋 Next Steps (Phase 2.1):")
        print("   🔜 Character-level n-gram models (3-5 grams)")
        print("   🔜 Statistical error detection systems")
        print("   🔜 Syllable context models and validation")
        print("   🔜 Enhanced frequency-based validation")
        
        # Save frequency model if analyzer was trained
        if analyzer and analyzer.is_trained:
            print(f"\n💾 Saving frequency model...")
            output_dir = Path("output/phase14")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = output_dir / "syllable_frequency_model"
            analyzer.save_frequency_data(str(model_path))
            print(f"   Model saved to: {model_path}.json/.pkl")
            
    except FileNotFoundError:
        print(f"❌ Error: Directory not found: {data_directory}")
        print("   Using sample texts for demonstration...")
        # Could add fallback demo here with hardcoded texts
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main() 