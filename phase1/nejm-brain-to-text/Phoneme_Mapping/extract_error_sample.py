"""extract_error_sample.py

This script extracts a sample of trials from the t15_copyTask.pkl file,
focusing on trials with phoneme errors, and saves them in a format
suitable for accent/dialect analysis.

Usage:
    python extract_error_sample.py --pkl /path/to/t15_copyTask.pkl --output error_sample.json

Output:
    A JSON file containing trial data with word-level alignment for error analysis.
"""

import argparse
import json
import pickle
import numpy as np
from collections import defaultdict


def calculate_error_rate(r, h):
    """Calculate edit distance between two sequences."""
    d = np.zeros((len(r)+1, len(h)+1), dtype=np.uint16)
    for i in range(len(r)+1):
        d[i, 0] = i
    for j in range(len(h)+1):
        d[0, j] = j
    
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j-1] + 1,  # substitution
                              d[i, j-1] + 1,     # insertion
                              d[i-1, j] + 1)     # deletion
    return d[len(r), len(h)]


def get_alignment(r, h):
    """
    Get the alignment between reference and hypothesis sequences.
    Returns a list of tuples: (operation, ref_item, hyp_item)
    Operations: 'match', 'substitute', 'insert', 'delete'
    """
    m, n = len(r), len(h)
    d = np.zeros((m+1, n+1), dtype=np.uint16)
    
    for i in range(m+1):
        d[i, 0] = i
    for j in range(n+1):
        d[0, j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if r[i-1] == h[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j-1] + 1, d[i, j-1] + 1, d[i-1, j] + 1)
    
    # Backtrace to get alignment
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and r[i-1] == h[j-1]:
            alignment.append(('match', r[i-1], h[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i, j] == d[i-1, j-1] + 1:
            alignment.append(('substitute', r[i-1], h[j-1]))
            i -= 1
            j -= 1
        elif j > 0 and d[i, j] == d[i, j-1] + 1:
            alignment.append(('insert', None, h[j-1]))
            j -= 1
        else:
            alignment.append(('delete', r[i-1], None))
            i -= 1
    
    return list(reversed(alignment))


def split_phonemes_by_word(phonemes, words):
    """
    Split a phoneme sequence into word-level chunks using SIL as delimiter.
    Returns a list of (word, phoneme_list) tuples.
    """
    word_phonemes = []
    current_phonemes = []
    word_idx = 0
    
    for p in phonemes:
        if p == 'SIL':
            if current_phonemes and word_idx < len(words):
                word_phonemes.append((words[word_idx], current_phonemes))
                word_idx += 1
            current_phonemes = []
        else:
            current_phonemes.append(p)
    
    # Handle last word if no trailing SIL
    if current_phonemes and word_idx < len(words):
        word_phonemes.append((words[word_idx], current_phonemes))
    
    return word_phonemes


def extract_sample(pkl_path, output_path, max_trials=100, include_correct=False):
    """
    Extract a sample of trials with error information.
    
    Args:
        pkl_path: Path to the pickle file
        output_path: Path for output JSON
        max_trials: Maximum number of error trials to extract
        include_correct: Whether to also include some correct trials for comparison
    """
    print(f"Loading pickle file: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        dat = pickle.load(f)
    
    # First, let's understand the data structure
    print("\n=== Data Structure ===")
    print(f"Keys in pickle: {list(dat.keys())}")
    print(f"Number of trials: {len(dat['cue_sentence'])}")
    
    # Sample a few entries to understand format
    print("\n=== Sample Entry (trial 0) ===")
    for key in dat.keys():
        val = dat[key][0]
        if isinstance(val, np.ndarray):
            print(f"  {key}: ndarray shape {val.shape}")
        elif isinstance(val, list):
            print(f"  {key}: list length {len(val)}, first few: {val[:5]}")
        else:
            print(f"  {key}: {type(val).__name__} = {val}")
    
    # Compute errors for each trial
    print("\n=== Computing errors ===")
    trials_with_errors = []
    error_counts = defaultdict(int)
    
    n_trials = len(dat['cue_sentence'])
    
    for i in range(n_trials):
        cue_phonemes = dat['cue_sentence_phonemes'][i]
        decoded_phonemes = dat['decoded_phonemes_raw'][i]
        
        # Filter out SIL for error calculation
        cue_no_sil = [p for p in cue_phonemes if p != 'SIL']
        dec_no_sil = [p for p in decoded_phonemes if p != 'SIL']
        
        n_errors = calculate_error_rate(cue_no_sil, dec_no_sil)
        
        if n_errors > 0:
            alignment = get_alignment(cue_no_sil, dec_no_sil)
            
            # Count substitution patterns
            for op, ref, hyp in alignment:
                if op == 'substitute':
                    error_counts[(ref, hyp)] += 1
            
            trials_with_errors.append({
                'trial_idx': i,
                'n_errors': int(n_errors),
                'n_phonemes': len(cue_no_sil),
                'error_rate': n_errors / len(cue_no_sil) if cue_no_sil else 0
            })
    
    print(f"Trials with errors: {len(trials_with_errors)} / {n_trials}")
    
    # Sort by number of errors and take a sample
    trials_with_errors.sort(key=lambda x: x['n_errors'], reverse=True)
    sample_indices = [t['trial_idx'] for t in trials_with_errors[:max_trials]]
    
    # If including correct trials, add some
    if include_correct:
        all_indices = set(range(n_trials))
        error_indices = set(sample_indices)
        correct_indices = list(all_indices - error_indices)
        np.random.seed(42)
        correct_sample = np.random.choice(correct_indices, 
                                          min(20, len(correct_indices)), 
                                          replace=False)
        sample_indices.extend(correct_sample)
    
    # Extract detailed data for sampled trials
    print(f"\n=== Extracting {len(sample_indices)} trials ===")
    
    extracted_trials = []
    for idx in sample_indices:
        cue_sent = dat['cue_sentence'][idx]
        cue_phonemes = dat['cue_sentence_phonemes'][idx]
        decoded_phonemes = dat['decoded_phonemes_raw'][idx]
        decoded_sent = dat['decoded_sentence'][idx]
        
        words = cue_sent.split()
        
        # Get word-level phoneme breakdown
        cue_by_word = split_phonemes_by_word(cue_phonemes, words)
        
        # Get alignment
        cue_no_sil = [p for p in cue_phonemes if p != 'SIL']
        dec_no_sil = [p for p in decoded_phonemes if p != 'SIL']
        alignment = get_alignment(cue_no_sil, dec_no_sil)
        
        # Extract substitution errors
        substitutions = [(ref, hyp) for op, ref, hyp in alignment if op == 'substitute']
        insertions = [hyp for op, ref, hyp in alignment if op == 'insert']
        deletions = [ref for op, ref, hyp in alignment if op == 'delete']
        
        trial_data = {
            'trial_idx': int(idx),
            'cue_sentence': cue_sent,
            'cue_words': words,
            'cue_phonemes': cue_phonemes,
            'cue_phonemes_by_word': cue_by_word,
            'decoded_phonemes_raw': decoded_phonemes,
            'decoded_sentence': decoded_sent,
            'alignment': alignment,
            'substitutions': substitutions,
            'insertions': insertions,
            'deletions': deletions,
            'post_implant_day': int(dat['post_implant_day'][idx]),
            'vocab_size': int(dat['vocab_size'][idx])
        }
        
        extracted_trials.append(trial_data)
    
    # Compute summary statistics
    print("\n=== Top 20 Substitution Patterns ===")
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    for (ref, hyp), count in sorted_errors[:20]:
        print(f"  {ref} -> {hyp}: {count}")
    
    # Prepare output
    output_data = {
        'n_total_trials': n_trials,
        'n_trials_with_errors': len(trials_with_errors),
        'n_extracted_trials': len(extracted_trials),
        'substitution_counts': {f"{ref}->{hyp}": count for (ref, hyp), count in sorted_errors},
        'trials': extracted_trials
    }
    
    # Save to JSON
    print(f"\n=== Saving to {output_path} ===")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Also save a smaller summary file
    summary_path = output_path.replace('.json', '_summary.json')
    summary_data = {
        'n_total_trials': n_trials,
        'n_trials_with_errors': len(trials_with_errors),
        'substitution_counts': {f"{ref}->{hyp}": count for (ref, hyp), count in sorted_errors},
        'top_substitutions': sorted_errors[:50],
        'sample_trials_preview': [
            {
                'cue': t['cue_sentence'],
                'decoded': t['decoded_sentence'],
                'substitutions': t['substitutions'][:5] if t['substitutions'] else []
            }
            for t in extracted_trials[:10]
        ]
    }
    
    print(f"=== Saving summary to {summary_path} ===")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print("\nDone! You can upload the JSON files to Claude for analysis.")
    print(f"  - Full sample: {output_path}")
    print(f"  - Summary only: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract error sample from pickle file')
    parser.add_argument('--pkl', required=True, help='Path to t15_copyTask.pkl')
    parser.add_argument('--output', default='error_sample.json', help='Output JSON path')
    parser.add_argument('--max-trials', type=int, default=100, help='Max error trials to extract')
    parser.add_argument('--include-correct', action='store_true', help='Include some correct trials')
    
    args = parser.parse_args()
    extract_sample(args.pkl, args.output, args.max_trials, args.include_correct)
