# -*- coding: utf-8 -*-
"""
Paragraph-by-paragraph analysis of Khmer syllable segmentation differences.
Goes through each chunk, splits into paragraphs, finds first difference, and moves on.
This allows step-by-step fixing of inconsistencies.
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

class ParagraphAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.problem_paragraphs = []
        
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

    def split_into_paragraphs(self, text):
        """Split text into paragraphs using various delimiters"""
        # Split by double newlines first (standard paragraph separator)
        paragraphs = text.split('\n\n')
        
        # Further split by single newlines if paragraphs are still very long
        refined_paragraphs = []
        for para in paragraphs:
            if len(para) > 300:  # If paragraph is too long, split by single newlines
                sub_paras = para.split('\n')
                refined_paragraphs.extend([p.strip() for p in sub_paras if p.strip()])
            else:
                if para.strip():
                    refined_paragraphs.append(para.strip())
        
        # Filter out very short paragraphs (less than 10 characters)
        meaningful_paragraphs = [p for p in refined_paragraphs if len(p) >= 3]
        
        return meaningful_paragraphs

    def test_paragraph(self, paragraph):
        """Test all segmentation methods on a single paragraph"""
        test_methods = [
            # Core syllable functions
            ('regex_basic', lambda t: khmer_syllables(t)),
            ('regex_advanced', lambda t: khmer_syllables_advanced(t)),
            ('non_regex_original', lambda t: khmer_syllables_no_regex(t)),
            ('non_regex_fast', lambda t: khmer_syllables_no_regex_fast(t)),
            
            # Paragraph functions (converted to list format)
            ('paragraph_basic', lambda t: segment_paragraph_to_subwords(t, separator='|').split('|') if segment_paragraph_to_subwords(t, separator='|') else []),
            ('paragraph_advanced', lambda t: segment_paragraph_to_subwords(t, method="advanced", separator='|').split('|') if segment_paragraph_to_subwords(t, method="advanced", separator='|') else []),
            ('paragraph_non_regex', lambda t: segment_paragraph_to_subwords_no_regex(t, separator='|').split('|') if segment_paragraph_to_subwords_no_regex(t, separator='|') else []),
            ('paragraph_non_regex_fast', lambda t: segment_paragraph_to_subwords_no_regex_fast(t, separator='|').split('|') if segment_paragraph_to_subwords_no_regex_fast(t, separator='|') else []),
            ('paragraph_optimized', lambda t: segment_paragraph_to_subwords_optimized(t, separator='|').split('|') if segment_paragraph_to_subwords_optimized(t, separator='|') else []),
        ]
        
        results = {}
        
        for name, func in test_methods:
            try:
                result = func(paragraph)
                
                # Ensure result is a list
                if isinstance(result, str):
                    result = result.split('|') if result else []
                
                results[name] = {
                    'result': result,
                    'syllable_count': len(result),
                    'result_hash': hashlib.md5('|'.join(result).encode('utf-8')).hexdigest()
                }
                
            except Exception as e:
                results[name] = {
                    'error': str(e),
                    'result': None,
                    'syllable_count': 0,
                    'result_hash': None
                }
                
        return results

    def check_paragraph_consistency(self, results):
        """Check if all methods produce the same result for a paragraph"""
        # Get all unique hashes (excluding errors)
        hashes = set()
        for method, data in results.items():
            if data.get('result_hash'):
                hashes.add(data['result_hash'])
        
        return len(hashes) <= 1  # Consistent if 0 or 1 unique hash

    def analyze_paragraph_differences(self, results, paragraph, chunk_id, paragraph_id):
        """Analyze and format differences for a problematic paragraph"""
        # Group methods by result hash
        hash_to_methods = defaultdict(list)
        for method, data in results.items():
            if data.get('result_hash'):
                hash_to_methods[data['result_hash']].append(method)
        
        problem_info = {
            'chunk_id': chunk_id,
            'paragraph_id': paragraph_id,
            'paragraph_text': paragraph,
            'paragraph_length': len(paragraph),
            'unique_results': len(hash_to_methods),
            'method_groups': dict(hash_to_methods),
            'detailed_results': {}
        }
        
        # Add detailed results for each method
        for method, data in results.items():
            if 'error' not in data:
                problem_info['detailed_results'][method] = {
                    'syllable_count': data['syllable_count'],
                    'first_10_syllables': data['result'][:10] if data['result'] else [],
                    'full_result': data['result'],
                    'result_hash': data['result_hash']
                }
            else:
                problem_info['detailed_results'][method] = {
                    'error': data['error']
                }
        
        return problem_info

    def run_paragraph_analysis(self, max_chunks=None):
        """Run paragraph-by-paragraph analysis"""
        print("=== PARAGRAPH-BY-PARAGRAPH KHMER SYLLABLE ANALYSIS ===")
        print(f"File: {self.file_path}")
        print("Strategy: Find first problematic paragraph in each chunk, then move to next chunk")
        print()
        
        # Read file in chunks
        chunks = self.read_file_in_chunks()
        
        if max_chunks:
            chunks = chunks[:max_chunks]
            print(f"Limiting analysis to first {max_chunks} chunks")
        
        print(f"Analyzing {len(chunks)} chunks...\n")
        
        total_chunks_with_problems = 0
        total_paragraphs_tested = 0
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            print(f"📁 CHUNK {chunk_idx + 1}/{len(chunks)}")
            print(f"   Length: {len(chunk)} characters")
            
            # Split chunk into paragraphs
            paragraphs = self.split_into_paragraphs(chunk)
            print(f"   Paragraphs found: {len(paragraphs)}")
            
            chunk_has_problem = False
            
            # Test each paragraph until we find a difference
            for para_idx, paragraph in enumerate(paragraphs):
                total_paragraphs_tested += 1
                
                print(f"   📝 Testing paragraph {para_idx + 1}: ", end="")
                
                # Test the paragraph
                results = self.test_paragraph(paragraph)
                
                # Check consistency
                is_consistent = self.check_paragraph_consistency(results)
                
                if not is_consistent:
                    print("❌ DIFFERENCES FOUND!")
                    
                    # Analyze the differences
                    problem_info = self.analyze_paragraph_differences(
                        results, paragraph, chunk_idx, para_idx
                    )
                    self.problem_paragraphs.append(problem_info)
                    
                    total_chunks_with_problems += 1
                    chunk_has_problem = True
                    
                    print(f"      📊 {problem_info['unique_results']} different result patterns")
                    print(f"      📏 Paragraph length: {len(paragraph)} characters")
                    print(f"      💾 Saved for analysis")
                    
                    # Skip remaining paragraphs in this chunk
                    if para_idx + 1 < len(paragraphs):
                        print(f"   ⏭️  Skipping remaining {len(paragraphs) - para_idx - 1} paragraphs in this chunk")
                    break
                else:
                    print("✅ OK")
            
            if not chunk_has_problem:
                print("   🎉 All paragraphs in this chunk are consistent!")
            
            print()  # Empty line between chunks
        
        # Summary
        print("=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total chunks analyzed: {len(chunks)}")
        print(f"Chunks with problems: {total_chunks_with_problems}")
        print(f"Chunks consistent: {len(chunks) - total_chunks_with_problems}")
        print(f"Total paragraphs tested: {total_paragraphs_tested}")
        print(f"Problematic paragraphs found: {len(self.problem_paragraphs)}")
        print(f"Success rate: {((len(chunks) - total_chunks_with_problems) / len(chunks) * 100):.1f}%")
        
        return len(chunks), total_chunks_with_problems

    def save_problem_paragraphs(self, output_file='problem_paragraphs.txt'):
        """Save problematic paragraphs to file for step-by-step fixing"""
        if not self.problem_paragraphs:
            print("✅ No problem paragraphs found - all are consistent!")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== PROBLEMATIC PARAGRAPHS FOR STEP-BY-STEP FIXING ===\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total problem paragraphs: {len(self.problem_paragraphs)}\n")
            f.write(f"Strategy: Fix these one by one until all methods are consistent\n\n")
            
            for i, problem in enumerate(self.problem_paragraphs, 1):
                f.write(f"{'='*80}\n")
                f.write(f"PROBLEM #{i} - Chunk {problem['chunk_id']}, Paragraph {problem['paragraph_id']}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"PARAGRAPH TEXT ({problem['paragraph_length']} chars):\n")
                f.write(f"{'-'*40}\n")
                f.write(f"{problem['paragraph_text']}\n")
                f.write(f"{'-'*40}\n\n")
                
                f.write(f"ANALYSIS:\n")
                f.write(f"- Number of different results: {problem['unique_results']}\n")
                f.write(f"- Text length: {problem['paragraph_length']} characters\n\n")
                
                f.write(f"METHOD GROUPINGS (methods producing same results):\n")
                for hash_val, methods in problem['method_groups'].items():
                    f.write(f"  Group {hash_val[:8]}...: {', '.join(methods)}\n")
                f.write("\n")
                
                f.write(f"SYLLABLE COUNTS BY METHOD:\n")
                for method, details in problem['detailed_results'].items():
                    if 'error' in details:
                        f.write(f"  {method:25}: ERROR - {details['error']}\n")
                    else:
                        f.write(f"  {method:25}: {details['syllable_count']:4d} syllables\n")
                f.write("\n")
                
                f.write(f"DETAILED RESULTS (first 20 syllables):\n")
                for method, details in problem['detailed_results'].items():
                    if 'error' not in details:
                        first_20 = details['first_10_syllables'][:20] if details['first_10_syllables'] else []
                        sample = ', '.join(f"'{s}'" for s in first_20)
                        f.write(f"  {method:25}: [{sample}]\n")
                f.write("\n")
                
                f.write(f"FULL RESULTS FOR DEBUGGING:\n")
                for method, details in problem['detailed_results'].items():
                    if 'error' not in details and details['full_result']:
                        f.write(f"\n{method}:\n")
                        result_str = ' | '.join(details['full_result'])
                        f.write(f"{result_str}\n")
                
                f.write(f"\n{'='*80}\n\n")
        
        print(f"📄 Problem paragraphs saved to: {output_file}")
        print(f"💡 Fix these {len(self.problem_paragraphs)} paragraphs one by one to achieve consistency!")

def main():
    # Configuration
    FILE_PATH = r"D:\data\ML\KhmerText\km.txt"
    MAX_CHUNKS = 20  # Start with smaller number for focused analysis
    
    try:
        analyzer = ParagraphAnalyzer(FILE_PATH)
        
        # Run paragraph-by-paragraph analysis
        total_chunks, problematic_chunks = analyzer.run_paragraph_analysis(max_chunks=MAX_CHUNKS)
        
        # Save problematic paragraphs
        analyzer.save_problem_paragraphs('problem_paragraphs.txt')
        
        print("\n🎯 NEXT STEPS:")
        print("1. Open 'problem_paragraphs.txt' to see the first problematic paragraph in each chunk")
        print("2. Fix the segmentation logic for the first problem")
        print("3. Re-run this script to see if the fix worked")
        print("4. Repeat until all paragraphs are consistent")
        
        if problematic_chunks == 0:
            print("\n🎉 CONGRATULATIONS! All chunks have consistent paragraphs!")
        else:
            print(f"\n⚠️  Found {problematic_chunks} chunks with problematic paragraphs")
            print("   Focus on fixing the first few problems to make the biggest impact")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the file path is correct and the file exists.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 