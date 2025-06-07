# -*- coding: utf-8 -*-
"""
Extract all unique tokens from large Khmer text file using khmer_syllables method.
Saves all unique tokens to a file, one token per line.
"""

import os
import time
from collections import OrderedDict
from subword_cluster import khmer_syllables

def extract_unique_tokens(input_file_path, output_file_path, chunk_size=10000):
    """
    Extract all unique tokens from a large text file using khmer_syllables.
    
    Args:
        input_file_path (str): Path to input Khmer text file
        output_file_path (str): Path to output tokens file
        chunk_size (int): Number of lines to process at once
    """
    
    print(f"=== Khmer Token Extraction ===")
    print(f"Input file: {input_file_path}")
    print(f"Output file: {output_file_path}")
    print(f"Processing in chunks of {chunk_size} lines...")
    print()
    
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found: {input_file_path}")
        return False
    
    # Use OrderedDict to maintain insertion order while ensuring uniqueness
    unique_tokens = OrderedDict()
    total_lines = 0
    total_tokens = 0
    
    start_time = time.time()
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            print("Reading and processing file...")
            
            lines_buffer = []
            
            for line_num, line in enumerate(infile, 1):
                lines_buffer.append(line.strip())
                
                # Process chunk when buffer is full
                if len(lines_buffer) >= chunk_size:
                    chunk_tokens = process_chunk(lines_buffer, line_num - len(lines_buffer) + 1, line_num)
                    total_tokens += len(chunk_tokens)
                    
                    # Add tokens to unique set
                    for token in chunk_tokens:
                        if token and token not in unique_tokens:  # Skip empty tokens
                            unique_tokens[token] = True
                    
                    total_lines += len(lines_buffer)
                    lines_buffer = []
                    
                    # Progress update
                    if line_num % (chunk_size * 10) == 0:
                        elapsed = time.time() - start_time
                        print(f"  Processed {line_num:,} lines, found {len(unique_tokens):,} unique tokens ({elapsed:.1f}s)")
            
            # Process remaining lines
            if lines_buffer:
                chunk_tokens = process_chunk(lines_buffer, total_lines + 1, total_lines + len(lines_buffer))
                total_tokens += len(chunk_tokens)
                
                for token in chunk_tokens:
                    if token and token not in unique_tokens:
                        unique_tokens[token] = True
                
                total_lines += len(lines_buffer)
        
        processing_time = time.time() - start_time
        
        print(f"\n=== Processing Complete ===")
        print(f"Total lines processed: {total_lines:,}")
        print(f"Total tokens found: {total_tokens:,}")
        print(f"Unique tokens: {len(unique_tokens):,}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Rate: {total_lines/processing_time:.0f} lines/second")
        print()
        
        # Save unique tokens to file
        print("Saving unique tokens to file...")
        save_start = time.time()
        
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for token in unique_tokens.keys():
                outfile.write(token + '\n')
        
        save_time = time.time() - save_start
        
        print(f"✅ Successfully saved {len(unique_tokens):,} unique tokens to: {output_file_path}")
        print(f"Save time: {save_time:.2f} seconds")
        
        # Show sample tokens
        sample_tokens = list(unique_tokens.keys())[:20]
        print(f"\nSample tokens (first 20):")
        for i, token in enumerate(sample_tokens, 1):
            print(f"  {i:2d}. '{token}'")
        
        if len(unique_tokens) > 20:
            print(f"  ... and {len(unique_tokens) - 20:,} more tokens")
        
        return True
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return False

def process_chunk(lines, start_line, end_line):
    """
    Process a chunk of lines and extract tokens using khmer_syllables.
    
    Args:
        lines (list): List of text lines to process
        start_line (int): Starting line number for progress tracking
        end_line (int): Ending line number for progress tracking
    
    Returns:
        list: List of all tokens found in the chunk
    """
    all_tokens = []
    
    for line in lines:
        if line.strip():  # Skip empty lines
            try:
                # Use khmer_syllables to tokenize the line
                tokens = khmer_syllables(line)
                all_tokens.extend(tokens)
            except Exception as e:
                print(f"Warning: Error processing line {start_line}: {e}")
                continue
    
    return all_tokens

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

if __name__ == "__main__":
    # File paths
    input_file = r"D:\data\ML\KhmerText\km.txt"
    output_file = "khmer_unique_tokens.txt"
    
    # Get file information
    get_file_info(input_file)
    
    # Extract tokens
    success = extract_unique_tokens(input_file, output_file, chunk_size=1000)
    
    if success:
        print(f"\n🎉 Token extraction completed successfully!")
        print(f"📁 Output file: {os.path.abspath(output_file)}")
    else:
        print(f"\n❌ Token extraction failed!") 