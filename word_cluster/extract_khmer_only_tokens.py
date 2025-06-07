# -*- coding: utf-8 -*-
"""
Extract unique KHMER-ONLY tokens from large Khmer text file using khmer_syllables method.
Filters out all non-Khmer tokens (spaces, punctuation, numbers, Latin characters).
Saves only pure Khmer tokens to a file, one token per line.
"""

import os
import time
from collections import OrderedDict
from subword_cluster import khmer_syllables, is_khmer_character

def is_pure_khmer_token(token):
    """
    Check if a token contains only Khmer characters.
    
    Args:
        token (str): Token to check
        
    Returns:
        bool: True if token contains only Khmer characters, False otherwise
    """
    if not token or not token.strip():
        return False
    
    # Check if all characters in the token are Khmer characters
    for char in token:
        if not is_khmer_character(char):
            return False
    
    return True

def extract_khmer_only_tokens(input_file_path, output_file_path, chunk_size=10000):
    """
    Extract all unique KHMER-ONLY tokens from a large text file using khmer_syllables.
    Filters out spaces, punctuation, numbers, Latin characters, and mixed tokens.
    
    Args:
        input_file_path (str): Path to input Khmer text file
        output_file_path (str): Path to output tokens file
        chunk_size (int): Number of lines to process at once
    """
    
    print(f"=== Khmer-Only Token Extraction ===")
    print(f"Input file: {input_file_path}")
    print(f"Output file: {output_file_path}")
    print(f"Processing in chunks of {chunk_size} lines...")
    print(f"Filter: ONLY pure Khmer tokens (no spaces, punctuation, numbers, Latin)")
    print()
    
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found: {input_file_path}")
        return False
    
    # Use OrderedDict to maintain insertion order while ensuring uniqueness
    unique_khmer_tokens = OrderedDict()
    total_lines = 0
    total_tokens = 0
    total_khmer_tokens = 0
    
    start_time = time.time()
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            print("Reading and processing file...")
            
            lines_buffer = []
            
            for line_num, line in enumerate(infile, 1):
                lines_buffer.append(line.strip())
                
                # Process chunk when buffer is full
                if len(lines_buffer) >= chunk_size:
                    chunk_tokens, chunk_khmer_tokens = process_chunk_khmer_only(lines_buffer, line_num - len(lines_buffer) + 1, line_num)
                    total_tokens += len(chunk_tokens)
                    total_khmer_tokens += len(chunk_khmer_tokens)
                    
                    # Add Khmer tokens to unique set
                    for token in chunk_khmer_tokens:
                        if token not in unique_khmer_tokens:
                            unique_khmer_tokens[token] = True
                    
                    total_lines += len(lines_buffer)
                    lines_buffer = []
                    
                    # Progress update
                    if line_num % (chunk_size * 10) == 0:
                        elapsed = time.time() - start_time
                        khmer_ratio = (total_khmer_tokens / total_tokens * 100) if total_tokens > 0 else 0
                        print(f"  Processed {line_num:,} lines, found {len(unique_khmer_tokens):,} unique Khmer tokens ({khmer_ratio:.1f}% Khmer, {elapsed:.1f}s)")
            
            # Process remaining lines
            if lines_buffer:
                chunk_tokens, chunk_khmer_tokens = process_chunk_khmer_only(lines_buffer, total_lines + 1, total_lines + len(lines_buffer))
                total_tokens += len(chunk_tokens)
                total_khmer_tokens += len(chunk_khmer_tokens)
                
                for token in chunk_khmer_tokens:
                    if token not in unique_khmer_tokens:
                        unique_khmer_tokens[token] = True
                
                total_lines += len(lines_buffer)
        
        processing_time = time.time() - start_time
        
        print(f"\n=== Processing Complete ===")
        print(f"Total lines processed: {total_lines:,}")
        print(f"Total tokens found: {total_tokens:,}")
        print(f"Total Khmer tokens: {total_khmer_tokens:,}")
        print(f"Unique Khmer tokens: {len(unique_khmer_tokens):,}")
        print(f"Khmer token ratio: {(total_khmer_tokens/total_tokens*100):.2f}%")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Rate: {total_lines/processing_time:.0f} lines/second")
        print()
        
        # Save unique Khmer tokens to file
        print("Saving unique Khmer tokens to file...")
        save_start = time.time()
        
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for token in unique_khmer_tokens.keys():
                outfile.write(token + '\n')
        
        save_time = time.time() - save_start
        
        print(f"✅ Successfully saved {len(unique_khmer_tokens):,} unique Khmer tokens to: {output_file_path}")
        print(f"Save time: {save_time:.2f} seconds")
        
        # Show sample tokens
        sample_tokens = list(unique_khmer_tokens.keys())[:20]
        print(f"\nSample Khmer tokens (first 20):")
        for i, token in enumerate(sample_tokens, 1):
            print(f"  {i:2d}. '{token}'")
        
        if len(unique_khmer_tokens) > 20:
            print(f"  ... and {len(unique_khmer_tokens) - 20:,} more Khmer tokens")
        
        # Show filtering statistics
        filtered_out = total_tokens - total_khmer_tokens
        print(f"\nFiltering Statistics:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Khmer tokens kept: {total_khmer_tokens:,}")
        print(f"  Non-Khmer tokens filtered out: {filtered_out:,}")
        print(f"  Filtering efficiency: {(filtered_out/total_tokens*100):.1f}% removed")
        
        return True
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return False

def process_chunk_khmer_only(lines, start_line, end_line):
    """
    Process a chunk of lines and extract only Khmer tokens using khmer_syllables.
    
    Args:
        lines (list): List of text lines to process
        start_line (int): Starting line number for progress tracking
        end_line (int): Ending line number for progress tracking
    
    Returns:
        tuple: (all_tokens, khmer_only_tokens) - lists of tokens
    """
    all_tokens = []
    khmer_only_tokens = []
    
    for line in lines:
        if line.strip():  # Skip empty lines
            try:
                # Use khmer_syllables to tokenize the line
                tokens = khmer_syllables(line)
                all_tokens.extend(tokens)
                
                # Filter for Khmer-only tokens
                for token in tokens:
                    if is_pure_khmer_token(token):
                        khmer_only_tokens.append(token)
                        
            except Exception as e:
                print(f"Warning: Error processing line {start_line}: {e}")
                continue
    
    return all_tokens, khmer_only_tokens

def get_file_info(file_path):
    """Get basic information about the input file."""
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        print(f"File info:")
        print(f"  Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"  Lines: {line_count:,}")
        print()
        
        return line_count, file_size
    except Exception as e:
        print(f"Warning: Could not get file info: {e}")
        return None, None

def analyze_token_sample(input_file_path, sample_lines=100):
    """
    Analyze a sample of the file to show the difference between all tokens and Khmer-only tokens.
    """
    print(f"=== Token Analysis Sample ({sample_lines} lines) ===")
    
    try:
        all_tokens = []
        khmer_tokens = []
        
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break
                if line.strip():
                    tokens = khmer_syllables(line.strip())
                    all_tokens.extend(tokens)
                    
                    for token in tokens:
                        if is_pure_khmer_token(token):
                            khmer_tokens.append(token)
        
        print(f"Sample analysis from first {sample_lines} lines:")
        print(f"  All tokens: {len(all_tokens)}")
        print(f"  Khmer-only tokens: {len(khmer_tokens)}")
        print(f"  Khmer ratio: {(len(khmer_tokens)/len(all_tokens)*100):.1f}%")
        
        # Show examples of filtered out tokens
        non_khmer_tokens = [token for token in all_tokens if not is_pure_khmer_token(token)]
        unique_non_khmer = list(OrderedDict.fromkeys(non_khmer_tokens))[:10]
        
        print(f"\nExamples of NON-Khmer tokens that will be filtered out:")
        for i, token in enumerate(unique_non_khmer, 1):
            print(f"  {i}. '{token}' (len={len(token)})")
        
        # Show examples of Khmer tokens that will be kept
        unique_khmer = list(OrderedDict.fromkeys(khmer_tokens))[:10]
        print(f"\nExamples of KHMER tokens that will be kept:")
        for i, token in enumerate(unique_khmer, 1):
            print(f"  {i}. '{token}' (len={len(token)})")
        
        print()
        
    except Exception as e:
        print(f"Error during sample analysis: {e}")

if __name__ == "__main__":
    # File paths
    input_file = r"D:\data\ML\KhmerText\km.txt"
    output_file = "khmer_only_unique_tokens.txt"
    
    # Get file information
    get_file_info(input_file)
    
    # Analyze a sample first
    analyze_token_sample(input_file, sample_lines=1000)
    
    # Extract Khmer-only tokens
    success = extract_khmer_only_tokens(input_file, output_file, chunk_size=1000)
    
    if success:
        print(f"\n🎉 Khmer-only token extraction completed successfully!")
        print(f"📁 Output file: {os.path.abspath(output_file)}")
        print(f"🔍 Contains only pure Khmer tokens (no spaces, punctuation, numbers, Latin)")
    else:
        print(f"\n❌ Khmer-only token extraction failed!") 