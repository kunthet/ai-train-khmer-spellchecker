# -*- coding: utf-8 -*-
"""Debug version of the generation to see what's happening"""

from generate_synthetic_khmer_tokens import KhmerTokenGenerator

def debug_generation():
    generator = KhmerTokenGenerator()
    
    print("=== Debug Full Generation ===")
    print(f"include_clusters=True, include_complex=True, include_multi_subscripts=True")
    print()
    
    # Call the full generation with debug
    tokens = generator.generate_all_tokens(include_clusters=True, include_complex=True, include_multi_subscripts=True)
    
    # Test specific patterns
    test_patterns = [
        'ស្ត្រី',        # woman  
        'សាស្ត្រ',       # science/subject
        'ហ្វ្រង្ក',      # French currency  
        'អង្ក្រង',       # k.o. large red ant
        'គញ្ច្រែង',      # user mentioned this should exist
        'ស្ត្រ',         # base pattern
        'ហ្វ្រ',         # base pattern
        'ង្ក្រ',         # base pattern
    ]
    
    print(f"\n=== Verification Check ===")
    print(f"Total tokens generated: {len(tokens)}")
    print(f"\nChecking for user-requested patterns:")
    
    found_count = 0
    for pattern in test_patterns:
        if pattern in tokens:
            print(f"  ✅ Found: {pattern}")
            found_count += 1
        else:
            print(f"  ❌ Missing: {pattern}")
    
    print(f"\nSuccess rate: {found_count}/{len(test_patterns)} = {found_count/len(test_patterns)*100:.1f}%")
    
    # Sample multi-subscript tokens from the final set
    multi_subscripts = [token for token in tokens if '្' in token and token.count('្') >= 2]
    print(f"\nMulti-subscript tokens found: {len(multi_subscripts)}")
    print("Sample multi-subscript tokens:")
    for i, token in enumerate(sorted(multi_subscripts)[:20]):
        print(f"  {i+1}. {token}")

if __name__ == "__main__":
    debug_generation() 