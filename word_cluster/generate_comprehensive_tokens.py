# -*- coding: utf-8 -*-
"""
Generate comprehensive synthetic Khmer tokens with all complex combinations.
"""

from generate_synthetic_khmer_tokens import KhmerTokenGenerator, save_synthetic_tokens, analyze_token_distribution

if __name__ == "__main__":
    generator = KhmerTokenGenerator()
    
    print("Generating COMPREHENSIVE Khmer token set...")
    print("This includes all complex combinations with vowels and diacritics.")
    print()
    
    # Generate comprehensive set
    tokens = generator.generate_all_tokens(include_clusters=True, include_complex=True)
    output_file = "synthetic_khmer_tokens_comprehensive.txt"
    
    # Save tokens
    save_synthetic_tokens(tokens, output_file, sort_by_length=True)
    
    # Analyze distribution
    analyze_token_distribution(tokens)
    
    print(f"\n🎉 Comprehensive synthetic Khmer token generation completed!")
    print(f"📁 Output: {output_file}")
    print(f"🧼 Ultra-clean vocabulary: {len(tokens):,} linguistically valid tokens")
    print(f"✨ Maximum coverage: All possible valid combinations") 