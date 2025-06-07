# -*- coding: utf-8 -*-
"""
Performance comparison between regex and non-regex Khmer syllable segmentation.
Updated to reflect actual performance characteristics and fix misleading labels.
Includes all four available implementations: basic regex, advanced regex, and two non-regex versions.
"""

import time
import statistics
from subword_cluster import (
    khmer_syllables, 
    khmer_syllables_advanced,
    khmer_syllables_no_regex, 
    khmer_syllables_no_regex_fast,
    segment_paragraph_to_subwords,
    segment_paragraph_to_subwords_no_regex,
    segment_paragraph_to_subwords_no_regex_fast,
    segment_paragraph_to_subwords_optimized
)

def generate_test_text(repeat_count=100):
    """Generate a large test text by repeating Khmer sentences"""
    base_text = "ខ្ញុំចង់រៀនភាសាខ្មែរ។ ក្រមុំលក់នំនៅភ្នំពេញ។ តើអ្នកជួយខ្ញុំបានទេ? អរគុណច្រើន! "
    return base_text * repeat_count

def benchmark_function(func, text, iterations=100):
    """Benchmark a function by running it multiple times and calculating average time"""
    times = []
    results = []
    
    for i in range(iterations):
        try:
            start_time = time.perf_counter()
            result = func(text)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            results.append(result)
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            return None
    
    # Verify all results are consistent
    first_result = results[0]
    for i, result in enumerate(results[1:], 2):
        if result != first_result:
            print(f"Warning: Inconsistent results between iterations 1 and {i}")
    
    return {
        'times': times,
        'average': statistics.mean(times),
        'min': min(times),
        'max': max(times),
        'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
        'result_length': len(first_result) if isinstance(first_result, list) else len(first_result.split('_'))
    }

def simple_speed_test():
    """Compare all four implementations with corrected labels"""
    print("=== Performance Speed Test ===")
    text = generate_test_text(1000)
    print(f"Processing {len(text):,} characters...")
    
    # Test all four versions with corrected variable names
    regex_basic_stats = benchmark_function(khmer_syllables, text)
    regex_advanced_stats = benchmark_function(khmer_syllables_advanced, text)
    basic_nonregex_stats = benchmark_function(khmer_syllables_no_regex, text)
    fast_nonregex_stats = benchmark_function(khmer_syllables_no_regex_fast, text)
    
    if not all([regex_basic_stats, regex_advanced_stats, basic_nonregex_stats, fast_nonregex_stats]):
        print("Error: One or more benchmarks failed!")
        return
    
    print(f"Basic regex version:               {regex_basic_stats['average']:.6f}s ± {regex_basic_stats['std_dev']:.6f}s")
    print(f"Advanced regex version:            {regex_advanced_stats['average']:.6f}s ± {regex_advanced_stats['std_dev']:.6f}s")
    print(f"Basic non-regex version:           {basic_nonregex_stats['average']:.6f}s ± {basic_nonregex_stats['std_dev']:.6f}s")
    print(f"Fast non-regex version:            {fast_nonregex_stats['average']:.6f}s ± {fast_nonregex_stats['std_dev']:.6f}s")
    print()
    
    # Find the fastest implementation as baseline
    all_stats = {
        'Basic Regex': regex_basic_stats,
        'Advanced Regex': regex_advanced_stats,
        'Basic Non-regex': basic_nonregex_stats,
        'Fast Non-regex': fast_nonregex_stats
    }
    
    fastest_time = min(stats['average'] for stats in all_stats.values())
    fastest_impl = [name for name, stats in all_stats.items() if stats['average'] == fastest_time][0]
    
    print(f"Performance relative to fastest implementation ({fastest_impl}):")
    for name, stats in all_stats.items():
        ratio = stats['average'] / fastest_time
        if ratio == 1.0:
            print(f"  {name:20}: {ratio:.2f}x (fastest)")
        else:
            print(f"  {name:20}: {ratio:.2f}x slower")
    print()
    
    # Accuracy verification
    print(f"Accuracy verification:")
    print(f"  Basic regex syllables:     {regex_basic_stats['result_length']}")
    print(f"  Advanced regex syllables:  {regex_advanced_stats['result_length']}")
    print(f"  Basic non-regex syllables: {basic_nonregex_stats['result_length']}")
    print(f"  Fast non-regex syllables:  {fast_nonregex_stats['result_length']}")
    
    all_counts_match = (regex_basic_stats['result_length'] == 
                       regex_advanced_stats['result_length'] == 
                       basic_nonregex_stats['result_length'] == 
                       fast_nonregex_stats['result_length'])
    print(f"  All syllable counts match: {all_counts_match}")

def accuracy_test():
    """Test accuracy of all implementations with detailed comparison"""
    print("\n=== Accuracy Test ===")
    test_sample = "ខ្ញុំចង់រៀនភាសាខ្មែរ។ ក្រមុំលក់នំនៅភ្នំពេញ។"
    
    try:
        # Test all four implementations
        basic_regex_syllables = khmer_syllables(test_sample)
        advanced_regex_syllables = khmer_syllables_advanced(test_sample)
        basic_nonregex_syllables = khmer_syllables_no_regex(test_sample)
        fast_nonregex_syllables = khmer_syllables_no_regex_fast(test_sample)
        
        # Convert to string format for easy comparison
        basic_regex_result = "|".join(basic_regex_syllables)
        advanced_regex_result = "|".join(advanced_regex_syllables)
        basic_nonregex_result = "|".join(basic_nonregex_syllables)
        fast_nonregex_result = "|".join(fast_nonregex_syllables)
        
        # print(f"Test text: {test_sample}")
        # print(f"Basic regex result:      {basic_regex_result}")
        # print(f"Advanced regex result:   {advanced_regex_result}")
        # print(f"Basic non-regex result:  {basic_nonregex_result}")
        # print(f"Fast non-regex result:   {fast_nonregex_result}")
        
        # Check for matches
        all_results = [basic_regex_result, advanced_regex_result, basic_nonregex_result, fast_nonregex_result]
        all_match = all(result == all_results[0] for result in all_results)
        print(f"All results match: {all_match}")
        
        if not all_match:
            print("\n⚠️  Differences found!")
            implementations = ["Basic regex", "Advanced regex", "Basic non-regex", "Fast non-regex"]
            results = [basic_regex_result, advanced_regex_result, basic_nonregex_result, fast_nonregex_result]
            
            for i in range(len(implementations)):
                for j in range(i+1, len(implementations)):
                    if results[i] != results[j]:
                        print(f"  {implementations[i]} vs {implementations[j]}: DIFFERENT")
        else:
            print("✅ All implementations produce identical results!")
            
    except Exception as e:
        print(f"Error during accuracy test: {e}")

def edge_case_test():
    """Test edge cases and special scenarios"""
    print("\n=== Edge Case Testing ===")
    
    test_cases = [
        ("", "Empty string"),
        ("ឲ្យ", "Independent vowel + coeng + consonant"),
        ("ឱ្យ", "Another independent vowel + coeng + consonant"),
        ("123 ABC", "Mixed numbers and Latin"),
        ("ខ្ញុំ   ចង់", "Multiple spaces"),
        ("ក្រុមការងារ។", "Complex syllables with punctuation")
    ]
    
    for test_text, description in test_cases:
        # print(f"\nTesting: {description}")
        # print(f"Input: '{test_text}'")
        
        try:
            basic_regex_result = khmer_syllables(test_text)
            advanced_regex_result = khmer_syllables_advanced(test_text)
            basic_nonregex_result = khmer_syllables_no_regex(test_text)
            fast_nonregex_result = khmer_syllables_no_regex_fast(test_text)
            
            # print(f"Basic regex:     {basic_regex_result}")
            # print(f"Advanced regex:  {advanced_regex_result}")
            # print(f"Basic non-regex: {basic_nonregex_result}")
            # print(f"Fast non-regex:  {fast_nonregex_result}")
            
            all_match = (basic_regex_result == advanced_regex_result == 
                        basic_nonregex_result == fast_nonregex_result)
            # print(f"All match: {all_match}")
            
            if not all_match:
                print("  ⚠️  Differences detected in this edge case!")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    simple_speed_test()
    accuracy_test()
    edge_case_test()
    
    # print("\n" + "="*70)
    # print("RECOMMENDATIONS:")
    # print()
    # print("🚀 BEST PERFORMANCE: Use khmer_syllables_advanced() (advanced regex)")
    # print("   • Fastest processing speed (~1.27x faster than basic regex)")
    # print("   • Identical accuracy to all other implementations")
    # print("   • Recommended for production use where maximum speed is needed")
    # print()
    # print("✅ STANDARD USE: Use khmer_syllables() (basic regex version)")
    # print("   • Slightly slower than advanced regex but still very fast")
    # print("   • Most thoroughly tested and documented")
    # print("   • Good default choice for most applications")
    # print()
    # print("⚡ NO REGEX DEPENDENCY: Use khmer_syllables_no_regex_fast()")
    # print("   • ~2x slower than regex versions")
    # print("   • No external regex dependency")
    # print("   • Good for dependency-constrained environments")
    # print()
    # print("📚 EDUCATIONAL: Use khmer_syllables_no_regex()")
    # print("   • Clearest algorithm implementation")
    # print("   • Slower performance (~3.6x slower than fastest)")
    # print("   • Good for understanding the segmentation logic")
    # print()
    # print("💡 PERFORMANCE SURPRISE: Advanced regex beats basic regex!")
    # print("💡 TIP: All implementations produce identical results!")
    # print("="*70) 