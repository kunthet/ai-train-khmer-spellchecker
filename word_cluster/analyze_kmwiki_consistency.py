#!/usr/bin/env python3
"""
Comprehensive analysis of Khmer Wikipedia data to test segmentation function consistency.

This script:
1. Reads each .txt file in the kmwiki_data directory
2. Reconstructs untokenized text (removes single spaces, keeps double spaces)
3. Tests all segmentation functions for consistency
4. Logs any differences found
"""

import os
import re
from pathlib import Path
from datetime import datetime
from subword_cluster import (
    khmer_syllables,
    khmer_syllables_advanced,
    khmer_syllables_no_regex,
    khmer_syllables_no_regex_fast
)

# Directory containing the Khmer Wikipedia data
KMWIKI_DIR = Path(r"D:\data\ML\KhmerText\seg_kmwiki_data\kmwiki_data")
LOG_FILE = "kmwiki_consistency_analysis.log"

def reconstruct_text(tokenized_text):
    """
    Reconstruct untokenized text by removing single spaces but keeping double spaces.
    
    Args:
        tokenized_text (str): Text with words separated by single spaces
        
    Returns:
        str: Reconstructed text with single spaces removed, double spaces preserved
    """
    # Replace double spaces with a placeholder to preserve them
    placeholder = "<<<DOUBLE_SPACE>>>"
    text = tokenized_text.replace("  ", placeholder)
    
    # Remove all single spaces
    text = text.replace(" ", "")
    
    # Restore double spaces
    text = text.replace(placeholder, "  ")
    
    return text

def test_function_consistency(text, filename):
    """
    Test all segmentation functions for consistency on the given text.
    
    Args:
        text (str): Text to test
        filename (str): Name of the file being tested
        
    Returns:
        tuple: (is_consistent, differences_dict)
    """
    methods = {
        'khmer_syllables': khmer_syllables,
        'khmer_syllables_advanced': khmer_syllables_advanced,
        'khmer_syllables_no_regex': khmer_syllables_no_regex,
        'khmer_syllables_no_regex_fast': khmer_syllables_no_regex_fast
    }
    
    results = {}
    
    # Run all methods
    for method_name, method_func in methods.items():
        try:
            result = method_func(text)
            results[method_name] = result
        except Exception as e:
            results[method_name] = f"ERROR: {e}"
    
    # Check for consistency
    valid_results = {k: v for k, v in results.items() if not str(v).startswith("ERROR:")}
    
    if len(valid_results) < 2:
        return False, {"error": "Too few valid results to compare"}
    
    # Compare all results
    result_values = list(valid_results.values())
    first_result = result_values[0]
    
    is_consistent = all(result == first_result for result in result_values)
    
    if not is_consistent:
        # Find which methods disagree
        differences = {}
        for method_name, result in valid_results.items():
            if result != first_result:
                differences[method_name] = {
                    'length': len(result),
                    'sample': result[:10],  # First 10 tokens
                    'differs_from_first': True
                }
        
        # Add first result for comparison
        first_method = list(valid_results.keys())[0]
        differences[first_method] = {
            'length': len(first_result),
            'sample': first_result[:10],
            'differs_from_first': False
        }
        
        return False, differences
    
    return True, {}

def analyze_file(filepath):
    """
    Analyze a single file for segmentation consistency.
    
    Args:
        filepath (Path): Path to the file to analyze
        
    Returns:
        dict: Analysis results
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            return {"status": "empty", "filename": filepath.name}
        
        # Reconstruct untokenized text
        untokenized_text = reconstruct_text(content)
        
        # Test consistency
        is_consistent, differences = test_function_consistency(untokenized_text, filepath.name)
        
        return {
            "status": "consistent" if is_consistent else "inconsistent",
            "filename": filepath.name,
            "file_size": filepath.stat().st_size,
            "original_text_length": len(content),
            "reconstructed_text_length": len(untokenized_text),
            "differences": differences,
            "sample_text": untokenized_text[:100] + "..." if len(untokenized_text) > 100 else untokenized_text
        }
        
    except Exception as e:
        return {
            "status": "error",
            "filename": filepath.name,
            "error": str(e)
        }

def main():
    """Main analysis function."""
    
    print("🧪 KMMER WIKIPEDIA DATA CONSISTENCY ANALYSIS")
    print("=" * 70)
    print(f"Directory: {KMWIKI_DIR}")
    print(f"Log file: {LOG_FILE}")
    print()
    
    if not KMWIKI_DIR.exists():
        print(f"❌ ERROR: Directory {KMWIKI_DIR} does not exist!")
        return
    
    # Get all .txt files
    txt_files = list(KMWIKI_DIR.glob("*.txt"))
    
    if not txt_files:
        print(f"❌ ERROR: No .txt files found in {KMWIKI_DIR}")
        return
    
    print(f"📁 Found {len(txt_files)} .txt files to analyze")
    print()
    
    # Initialize counters
    total_files = len(txt_files)
    consistent_files = 0
    inconsistent_files = 0
    error_files = 0
    
    # Open log file
    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
        # Write header
        log_file.write(f"KHMER WIKIPEDIA SEGMENTATION CONSISTENCY ANALYSIS\n")
        log_file.write(f"Analysis started: {datetime.now().isoformat()}\n")
        log_file.write(f"Directory: {KMWIKI_DIR}\n")
        log_file.write(f"Total files: {total_files}\n")
        log_file.write("=" * 80 + "\n\n")
        
        # Process each file
        for i, filepath in enumerate(txt_files, 1):
            print(f"📄 [{i:3d}/{total_files}] {filepath.name}")
            
            result = analyze_file(filepath)
            
            if result["status"] == "consistent":
                consistent_files += 1
                print(f"    ✅ CONSISTENT")
                
                # Log brief success
                # log_file.write(f"✅ CONSISTENT: {filepath.name}\n")
                # log_file.write(f"   File size: {result['file_size']:,} bytes\n")
                # log_file.write(f"   Text length: {result['reconstructed_text_length']:,} characters\n\n")
                
            elif result["status"] == "inconsistent":
                inconsistent_files += 1
                print(f"    ❌ INCONSISTENT - differences found!")
                
                # Log detailed inconsistency
                log_file.write(f"❌ INCONSISTENT: {filepath.name}\n")
                log_file.write(f"   File size: {result['file_size']:,} bytes\n")
                log_file.write(f"   Text length: {result['reconstructed_text_length']:,} characters\n")
                log_file.write(f"   Sample text: {result['sample_text']}\n")
                log_file.write("   DIFFERENCES:\n")
                
                for method_name, diff_info in result["differences"].items():
                    log_file.write(f"     {method_name}:\n")
                    log_file.write(f"       Length: {diff_info['length']}\n")
                    log_file.write(f"       Sample: {diff_info['sample']}\n")
                    log_file.write(f"       Differs: {diff_info['differs_from_first']}\n")
                
                log_file.write("\n" + "-" * 50 + "\n\n")
                
            else:  # error
                error_files += 1
                print(f"    ⚠️  ERROR: {result.get('error', 'Unknown error')}")
                
                # Log error
                log_file.write(f"⚠️ ERROR: {filepath.name}\n")
                log_file.write(f"   Error: {result.get('error', 'Unknown error')}\n\n")
        
        # Write summary
        print("\n" + "=" * 70)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Total files analyzed: {total_files}")
        print(f"Consistent files:     {consistent_files} ({consistent_files/total_files*100:.1f}%)")
        print(f"Inconsistent files:   {inconsistent_files} ({inconsistent_files/total_files*100:.1f}%)")
        print(f"Error files:          {error_files} ({error_files/total_files*100:.1f}%)")
        print()
        
        if inconsistent_files == 0:
            print("🎉 SUCCESS: All files show consistent segmentation across all methods!")
        else:
            print(f"⚠️  ATTENTION: {inconsistent_files} files show inconsistencies - check {LOG_FILE} for details")
        
        # Write summary to log
        log_file.write("=" * 80 + "\n")
        log_file.write("ANALYSIS SUMMARY\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Analysis completed: {datetime.now().isoformat()}\n")
        log_file.write(f"Total files analyzed: {total_files}\n")
        log_file.write(f"Consistent files:     {consistent_files} ({consistent_files/total_files*100:.1f}%)\n")
        log_file.write(f"Inconsistent files:   {inconsistent_files} ({inconsistent_files/total_files*100:.1f}%)\n")
        log_file.write(f"Error files:          {error_files} ({error_files/total_files*100:.1f}%)\n")
        
        if inconsistent_files == 0:
            log_file.write("\n🎉 SUCCESS: All files show consistent segmentation across all methods!\n")
        else:
            log_file.write(f"\n⚠️ ATTENTION: {inconsistent_files} files show inconsistencies.\n")

if __name__ == "__main__":
    main() 