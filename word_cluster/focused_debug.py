# -*- coding: utf-8 -*-
"""
Focused debugging of specific problem paragraphs to identify exact differences
"""

from subword_cluster import (
    khmer_syllables,
    khmer_syllables_no_regex,
    khmer_syllables_no_regex_fast, 
    khmer_syllables_advanced
)

def analyze_problem_paragraph(text, problem_id):
    """Analyze a specific problematic paragraph"""
    print(f"=== PROBLEM #{problem_id} ANALYSIS ===")
    print(f"Text: {text}")
    print(f"Length: {len(text)} characters")
    print()
    
    # Test the core functions
    methods = [
        ('regex_basic', khmer_syllables),
        ('regex_advanced', khmer_syllables_advanced),
        ('non_regex_original', khmer_syllables_no_regex),
        ('non_regex_fast', khmer_syllables_no_regex_fast)
    ]
    
    results = {}
    for name, func in methods:
        result = func(text)
        results[name] = result
        print(f"{name:20}: {len(result):3d} syllables")
    
    print()
    
    # Find differences by comparing results
    print("DETAILED DIFFERENCES:")
    max_len = max(len(r) for r in results.values())
    
    for i in range(max_len):
        syllables_at_pos = {}
        for method_name, result in results.items():
            if i < len(result):
                syllables_at_pos[method_name] = result[i]
            else:
                syllables_at_pos[method_name] = "END"
        
        # Check if all syllables at this position are the same
        unique_syllables = set(syllables_at_pos.values())
        if len(unique_syllables) > 1:
            print(f"Position {i:2d}:")
            for method, syllable in syllables_at_pos.items():
                print(f"  {method:20}: '{syllable}'")
            print()
    
    # Show full results for comparison
    print("FULL RESULTS:")
    for name, result in results.items():
        result_str = ' | '.join(result)
        print(f"{name}:")
        print(f"  {result_str}")
        print()
    
    return results

def main():
    # Problem paragraphs from the analysis
    problems = [
        {
            'id': 1,
            'text': 'វង់ភ្លេងអាយ៉ៃ កើតឡើងដោយបុរសឈ្មោះយ៉ៃ តាំងពីចុងសតវត្សទី១៩មកម្ល៉េះ - វិទ្យុវាយោ - VAYO FM Radioព័ត៌មានជាតិ'
        },
        {
            'id': 2,
            'text': 'អាយ៉ៃឆ្លងឆ្លើយ គឺជាការច្រៀងឆ្លងឆ្លើយកំណាព្យកាព្យឃ្លោងដោយឥតព្រាងទុក ។ ជួនកាល គេមានច្រៀងជាពីរគូ ឬបីគូដែរ ដូចអ្វីដែលលោក-អ្នក បានស្តាប់នូវទម្រង់ចម្រៀងអាយ៉ៃពីខាងដើមនោះឯង ។'
        },
        {
            'id': 3,
            'text': 'ជនភៀសខ្លួន ព្យាយាម​ឆ្លងទឹក​ស្ទឹង​នៅ​ក្បែរ​ព្រំដែន​​រវាង​ក្រិក​ និង​ម៉ាសេដ្វាន់ ថ្ងៃទី​១៤​ មីនា ២០១៦ Reuters/Alexandros Avramidisយ៉ាង​ណាមិញ ​បើយោង​តាម​ប្រសាសន៍​លោក​ស្រី​អង់ហ្គេឡា មែគែល ​កិច្ច​ព្រមព្រៀង​នេះ​ចូល​ជា​ធរមាន​នៅ​ថ្ងៃ​អាទិត្យ​នេះ​មែន ​ តែ​ការ​បញ្ជូន​ជនភៀសខ្លួន​ឲ្យ​ត្រលប់​ទៅ​តួកគី​គឺ​ត្រូវ​ចាប់ផ្តើម​នៅ​ថ្ងៃ​ ទី​៤​មេសា ​ឯនោះ​ទេ។'
        }
    ]
    
    for problem in problems:
        analyze_problem_paragraph(problem['text'], problem['id'])
        print("=" * 80)
        print()

if __name__ == "__main__":
    main() 