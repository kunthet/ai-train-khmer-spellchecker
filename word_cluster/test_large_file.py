# -*- coding: utf-8 -*-
"""
Comprehensive test to find differences between Khmer syllable segmentation implementations.
Focuses on identifying and reporting inconsistencies for debugging purposes.
"""

import os
import time
import hashlib
from collections import defaultdict
from subword_cluster import (
    khmer_syllables,
    khmer_syllables_no_regex,
    khmer_syllables_no_regex_fast, 
    khmer_syllables_advanced,
    segment_paragraph_to_subwords,
    segment_paragraph_to_subwords_no_regex,
    segment_paragraph_to_subwords_no_regex_fast,
    segment_paragraph_to_subwords_optimized
)

class KhmerDifferenceAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.differences = []
        self.consistent_chunks = []
        self.performance_stats = {}
        
    def read_file_in_chunks(self, chunk_size=1000):
        """Read file in chunks to handle large files efficiently"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        chunks = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split into lines and then into chunks
        lines = content.split('\n')
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk) + len(line) > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk += line + '\n' if current_chunk else line
                
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks

    def test_all_methods(self, text, chunk_id):
        """Test all segmentation methods on a text chunk"""
        # REMOVED: redundant zero-width space cleanup since functions now handle this internally
        
        test_methods = [
            # Syllable-level functions
            ('regex_basic', lambda t: khmer_syllables(t)),
            ('regex_advanced', lambda t: khmer_syllables_advanced(t)),
            ('non_regex_original', lambda t: khmer_syllables_no_regex(t)),
            ('non_regex_fast', lambda t: khmer_syllables_no_regex_fast(t)),
            
            # Paragraph-level functions
            ('paragraph_basic', lambda t: segment_paragraph_to_subwords(t, separator='|').split('|') if segment_paragraph_to_subwords(t, separator='|') else []),
            ('paragraph_advanced', lambda t: segment_paragraph_to_subwords(t, method="advanced", separator='|').split('|') if segment_paragraph_to_subwords(t, method="advanced", separator='|') else []),
            ('paragraph_non_regex', lambda t: segment_paragraph_to_subwords_no_regex(t, separator='|').split('|') if segment_paragraph_to_subwords_no_regex(t, separator='|') else []),
            ('paragraph_non_regex_fast', lambda t: segment_paragraph_to_subwords_no_regex_fast(t, separator='|').split('|') if segment_paragraph_to_subwords_no_regex_fast(t, separator='|') else []),
            ('paragraph_optimized', lambda t: segment_paragraph_to_subwords_optimized(t, separator='|').split('|') if segment_paragraph_to_subwords_optimized(t, separator='|') else []),
        ]
        
        results = {}
        
        for name, func in test_methods:
            try:
                start_time = time.perf_counter()
                result = func(text)
                end_time = time.perf_counter()
                
                # Ensure result is a list
                if isinstance(result, str):
                    result = result.split('|') if result else []
                
                results[name] = {
                    'result': result,
                    'time': end_time - start_time,
                    'syllable_count': len(result),
                    'result_hash': hashlib.md5('|'.join(result).encode('utf-8')).hexdigest()
                }
                
            except Exception as e:
                results[name] = {
                    'error': str(e),
                    'result': None,
                    'time': 0,
                    'syllable_count': 0,
                    'result_hash': None
                }
                
        return results

    def analyze_differences(self, results, chunk_id, text_sample):
        """Analyze results and return detailed difference information"""
        # Get all unique hashes
        hashes = set()
        hash_to_methods = defaultdict(list)
        
        for method, data in results.items():
            if data.get('result_hash'):
                hashes.add(data['result_hash'])
                hash_to_methods[data['result_hash']].append(method)
        
        # If more than one unique hash, we have differences
        if len(hashes) > 1:
            difference = {
                'chunk_id': chunk_id,
                'text_sample': text_sample[:100] + '...' if len(text_sample) > 100 else text_sample,
                'text_length': len(text_sample),
                'unique_result_count': len(hashes),
                'method_groups': dict(hash_to_methods),
                'detailed_analysis': {},
                'syllable_counts': {}
            }
            
            # Detailed analysis for each method
            for method, data in results.items():
                if 'error' not in data:
                    difference['detailed_analysis'][method] = {
                        'syllable_count': data['syllable_count'],
                        'first_10_syllables': data['result'][:10] if data['result'] else [],
                        'last_5_syllables': data['result'][-5:] if data['result'] and len(data['result']) > 5 else [],
                        'result_hash': data['result_hash']
                    }
                    difference['syllable_counts'][method] = data['syllable_count']
                else:
                    difference['detailed_analysis'][method] = {
                        'error': data['error']
                    }
            
            return difference
        
        return None

    def run_difference_analysis(self, max_chunks=None, save_consistent=False):
        """Run analysis focusing on finding differences"""
        print(f"=== Khmer Syllable Difference Analysis ===")
        print(f"File: {self.file_path}")
        
        # Read file in chunks
        print("Reading file in chunks...")
        chunks = self.read_file_in_chunks()
        
        if max_chunks:
            chunks = chunks[:max_chunks]
            print(f"Limiting analysis to first {max_chunks} chunks")
            
        print(f"Analyzing {len(chunks)} chunks for differences...")
        print()
        
        # Initialize tracking
        method_times = defaultdict(list)
        total_differences = 0
        
        # Test each chunk
        for i, chunk in enumerate(chunks, 1):
            if i % 10 == 0 or i <= 10:
                print(f"Processing chunk {i}/{len(chunks)}...", end=' ')
            
            # Test all methods
            results = self.test_all_methods(chunk, i-1)
            
            # Analyze for differences
            difference = self.analyze_differences(results, i-1, chunk)
            
            if difference:
                self.differences.append(difference)
                total_differences += 1
                if i % 10 == 0 or i <= 10:
                    print("DIFF!")
            else:
                if save_consistent:
                    self.consistent_chunks.append({
                        'chunk_id': i-1,
                        'text_sample': chunk[:100] + '...' if len(chunk) > 100 else chunk,
                        'syllable_count': list(results.values())[0]['syllable_count']
                    })
                if i % 10 == 0 or i <= 10:
                    print("OK")
            
            # Collect performance stats
            for method, data in results.items():
                if 'error' not in data:
                    method_times[method].append(data['time'])
        
        # Store performance stats
        self.performance_stats = {
            'method_times': dict(method_times),
            'total_chunks': len(chunks),
            'total_differences': total_differences,
            'consistency_rate': ((len(chunks) - total_differences) / len(chunks) * 100)
        }
        
        return len(chunks), total_differences

    def save_detailed_differences(self, output_file='detailed_differences.txt'):
        """Save comprehensive difference analysis to file"""
        if not self.differences:
            print("✅ No differences found - all methods produce consistent results!")
            return
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== KHMER SYLLABLE SEGMENTATION DIFFERENCE ANALYSIS ===\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total differences found: {len(self.differences)}\n")
            f.write(f"Consistency rate: {self.performance_stats.get('consistency_rate', 0):.2f}%\n")
            f.write(f"File analyzed: {self.file_path}\n\n")
            
            # Summary of most common difference patterns
            f.write("=== DIFFERENCE PATTERN SUMMARY ===\n")
            pattern_counts = defaultdict(int)
            for diff in self.differences:
                pattern_counts[diff['unique_result_count']] += 1
            
            for pattern_count, frequency in sorted(pattern_counts.items()):
                f.write(f"  {frequency} chunks had {pattern_count} different result patterns\n")
            f.write("\n")
            
            # Detailed differences
            f.write("=== DETAILED DIFFERENCES ===\n\n")
            
            for i, diff in enumerate(self.differences, 1):
                f.write(f"--- DIFFERENCE #{i} (Chunk {diff['chunk_id']}) ---\n")
                f.write(f"Text length: {diff['text_length']} characters\n")
                f.write(f"Text sample: {diff['text_sample']}\n")
                f.write(f"Number of different result patterns: {diff['unique_result_count']}\n\n")
                
                f.write("Method groupings (methods producing same results):\n")
                for hash_val, methods in diff['method_groups'].items():
                    f.write(f"  Group {hash_val[:8]}...: {', '.join(methods)}\n")
                f.write("\n")
                
                f.write("Syllable counts by method:\n")
                for method, count in sorted(diff['syllable_counts'].items()):
                    f.write(f"  {method:25}: {count:4d} syllables\n")
                f.write("\n")
                
                f.write("Sample syllables (first 10):\n")
                for method, details in diff['detailed_analysis'].items():
                    if 'error' in details:
                        f.write(f"  {method:25}: ERROR - {details['error']}\n")
                    else:
                        sample = ', '.join(f"'{s}'" for s in details['first_10_syllables'])
                        f.write(f"  {method:25}: [{sample}]\n")
                f.write("\n")
                
                f.write("="*80 + "\n\n")
        
        print(f"📄 Detailed differences saved to: {output_file}")

    def save_consistent_chunks(self, output_file='consistent_chunks.txt'):
        """Save information about chunks that are consistent across all methods"""
        if not self.consistent_chunks:
            print("No consistent chunks to save.")
            return
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== CONSISTENT CHUNKS (All methods agree) ===\n")
            f.write(f"Total consistent chunks: {len(self.consistent_chunks)}\n\n")
            
            for chunk in self.consistent_chunks:
                f.write(f"Chunk {chunk['chunk_id']}: {chunk['syllable_count']} syllables\n")
                f.write(f"Text: {chunk['text_sample']}\n\n")
        
        print(f"📄 Consistent chunks saved to: {output_file}")

    def generate_summary_report(self):
        """Generate a summary report of the analysis"""
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Total chunks analyzed: {self.performance_stats['total_chunks']}")
        print(f"Chunks with differences: {self.performance_stats['total_differences']}")
        print(f"Chunks consistent: {self.performance_stats['total_chunks'] - self.performance_stats['total_differences']}")
        print(f"Consistency rate: {self.performance_stats['consistency_rate']:.2f}%")
        print()
        
        if self.differences:
            print(f"📊 DIFFERENCE PATTERNS:")
            pattern_counts = defaultdict(int)
            for diff in self.differences:
                pattern_counts[diff['unique_result_count']] += 1
            
            for pattern_count, frequency in sorted(pattern_counts.items()):
                print(f"  {frequency:3d} chunks with {pattern_count} different result patterns")
        
        # Performance summary
        print(f"\n⚡ PERFORMANCE (fastest methods):")
        times = self.performance_stats['method_times']
        if times:
            fastest_time = min(sum(times[method])/len(times[method]) for method in times.keys())
            
            sorted_methods = sorted(times.keys(), 
                                  key=lambda m: sum(times[m])/len(times[m]))
            
            for method in sorted_methods[:3]:  # Top 3 fastest
                avg_time = sum(times[method]) / len(times[method])
                relative_speed = fastest_time / avg_time
                print(f"  {method:25}: {avg_time:.6f}s ({relative_speed:.2f}x)")

def main():
    # Configuration
    FILE_PATH = r"D:\data\ML\KhmerText\km.txt"
    MAX_CHUNKS = 50  # Limit for testing, set to None for full file
    
    try:
        analyzer = KhmerDifferenceAnalyzer(FILE_PATH)
        
        # Run difference analysis
        total_chunks, differences = analyzer.run_difference_analysis(
            max_chunks=MAX_CHUNKS, 
            save_consistent=True
        )
        
        # Generate reports
        analyzer.generate_summary_report()
        analyzer.save_detailed_differences('detailed_differences.txt')
        
        if analyzer.consistent_chunks:
            analyzer.save_consistent_chunks('consistent_chunks.txt')
        
        print(f"\n=== FILES GENERATED ===")
        print(f"📄 detailed_differences.txt - Complete difference analysis")
        if analyzer.consistent_chunks:
            print(f"📄 consistent_chunks.txt - Chunks where all methods agree")
        
        if differences == 0:
            print("\n🎉 All methods produce consistent results!")
        else:
            print(f"\n⚠️  Found {differences} chunks with differences - see detailed_differences.txt")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the file path is correct and the file exists.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 