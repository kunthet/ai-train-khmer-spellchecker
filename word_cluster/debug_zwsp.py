#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import subword_cluster
importlib.reload(subword_cluster)

from subword_cluster import (
    khmer_syllables,
    khmer_syllables_advanced,
    khmer_syllables_no_regex,
)

def test_zwsp_impact():
    # Sample text with zero-width space
    text_with_zwsp = 'Apple\u200bឲ្យប្រើ'
    text_without_zwsp = text_with_zwsp.replace('\u200b', '')
    
    methods = [
        ('Basic', khmer_syllables),
        ('Advanced', khmer_syllables_advanced),
        ('No-regex', khmer_syllables_no_regex),
    ]
    
    print('=== WITH Zero-Width Space ===')
    print(f'Text: {repr(text_with_zwsp)}')
    results_with = {}
    for name, func in methods:
        result = func(text_with_zwsp)
        results_with[name] = result
        print(f'{name:10}: {result} (count: {len(result)})')
    
    print('\n=== WITHOUT Zero-Width Space (after cleanup) ===')
    print(f'Text: {repr(text_without_zwsp)}')
    results_without = {}
    for name, func in methods:
        result = func(text_without_zwsp)
        results_without[name] = result
        print(f'{name:10}: {result} (count: {len(result)})')
    
    print('\n=== Analysis ===')
    print('With ZWSP consistency:', len(set(str(r) for r in results_with.values())) == 1)
    print('Without ZWSP consistency:', len(set(str(r) for r in results_without.values())) == 1)
    
    # Check what changes when ZWSP is removed
    print('\n=== Changes when ZWSP removed ===')
    for name in methods:
        method_name = name[0]
        before = results_with[method_name]
        after = results_without[method_name]
        if before != after:
            print(f'{method_name}: CHANGED')
            print(f'  Before: {before}')
            print(f'  After:  {after}')
        else:
            print(f'{method_name}: NO CHANGE')

if __name__ == "__main__":
    test_zwsp_impact() 