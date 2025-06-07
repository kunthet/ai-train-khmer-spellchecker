from analyze_kmwiki_consistency import reconstruct_text

# Test the text reconstruction
test_cases = [
    'ខ្ញុំ ចង់ រៀន ភាសា  ខ្មែរ  ឲ្យ បាន ល្អ',
    'នេះ គឺ ជា ការ សាកល្បង',
    'ពាក្យ មួយ  ពាក្យ ពីរ',
]

print("Testing text reconstruction logic:")
print("-" * 50)

for i, test in enumerate(test_cases, 1):
    result = reconstruct_text(test)
    print(f'Test {i}:')
    print(f'  Original: {repr(test)}')
    print(f'  Result:   {repr(result)}')
    print()

print("✅ Text reconstruction test completed!") 