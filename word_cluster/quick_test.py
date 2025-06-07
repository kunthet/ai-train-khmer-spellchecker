# -*- coding: utf-8 -*-
"""Quick test for specific patterns"""

from generate_synthetic_khmer_tokens import KhmerTokenGenerator

def quick_test():
    generator = KhmerTokenGenerator()
    
    # Test only the documented patterns
    documented = generator.generate_documented_specific_patterns()
    
    # Check for the specific user-requested pattern
    target = 'គញ្ច្រែង'
    
    if target in documented:
        print(f"✅ SUCCESS: Found {target} in documented patterns")
    else:
        print(f"❌ MISSING: {target} not found in documented patterns")
    
    print(f"\nDocumented patterns count: {len(documented)}")
    print("All documented patterns:")
    for i, pattern in enumerate(sorted(documented), 1):
        print(f"  {i:2d}. {pattern}")
    
    # Test also full generation for the specific pattern
    print(f"\nTesting full generation...")
    all_tokens = generator.generate_all_tokens(include_clusters=True, include_complex=True, include_multi_subscripts=True)
    
    if target in all_tokens:
        print(f"✅ SUCCESS: Found {target} in full token set")
    else:
        print(f"❌ MISSING: {target} not found in full token set")
    
    print(f"Total tokens: {len(all_tokens)}")

if __name__ == "__main__":
    quick_test() 