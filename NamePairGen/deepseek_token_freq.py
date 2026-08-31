"""
Token Frequency Analyzer
Analyzes token frequency distribution in a corpus using a trained tokenizer
"""

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict, Counter
import questionary
from questionary import Choice, Style

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# Constants
SEP_TOKEN = "<SEP>"
UNKNOWN_TOKEN = "<UNK>"
PADDING_TOKEN = "<PAD>"
BEGIN_OF_STREAM_TOKEN = "<BOS>"
END_OF_STREAM_TOKEN = "<EOS>"
RESPONSE_TOKEN = "<RESP>"
SPECIAL_TOKENS = [PADDING_TOKEN, BEGIN_OF_STREAM_TOKEN, END_OF_STREAM_TOKEN, UNKNOWN_TOKEN]


def get_folder_names(directory_path, sort=True, exclude_hidden=False):
    """Get folder names from a directory."""
    try:
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory '{directory_path}' not found.")
        
        if not path.is_dir():
            raise NotADirectoryError(f"'{directory_path}' is not a directory.")
        
        folders = []
        for item in path.iterdir():
            if item.is_dir():
                folder_name = item.name
                if exclude_hidden and folder_name.startswith('.'):
                    continue
                folders.append(folder_name)
        
        if sort:
            folders.sort()
        
        return folders
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def get_file_names(directory_path, sort=True, exclude_hidden=False):
    """Get file names from a directory."""
    try:
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory '{directory_path}' not found.")
        
        if not path.is_dir():
            raise NotADirectoryError(f"'{directory_path}' is not a directory.")
        
        files = []
        for item in path.iterdir():
            if item.is_file():
                file_name = item.name
                if exclude_hidden and file_name.startswith('.'):
                    continue
                files.append(file_name)
        
        if sort:
            files.sort()
        
        return files
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def train_tokenizer(vocab_size, special_tokens_param, unk_tok, added_toks, training_data, train=True):
    """Train BPE tokenizer."""
    bpe_trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens_param)
    tokenizer = Tokenizer(models.BPE(unk_token=unk_tok))
    tokenizer.add_tokens(added_toks)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    
    if train:
        tokenizer.train_from_iterator(training_data, trainer=bpe_trainer)
    
    return tokenizer


def prompt_select(message: str, choices: list) -> any:
    """Prompt user to select from choices."""
    return questionary.select(
        message,
        choices=choices,
        style=Style([("highlighted", "reverse")]),
    ).ask()


def load_corpus_data(data_path: str = 'data') -> Tuple[List[str], List[str]]:
    """Load all text data from the data directory."""
    folder_names = get_folder_names(data_path)
    
    class_tokens = [RESPONSE_TOKEN]
    corpus_data = []
    num_ent = 0
    
    print(f"\nLoading corpus from {data_path}...")
    for folder in tqdm(folder_names, desc='Processing folders'):
        files_in_folder = get_file_names(f'data/{folder}')
        
        for file in tqdm(files_in_folder, desc=f'  Processing {folder}', leave=False):
            ac_identifier = file.split("-names.txt")[0]
            class_tokens.append(f'<{ac_identifier}>')
                
            with open(f'data/{folder}/{file}', 'r', encoding='utf-8') as r_file:
                f_data = r_file.read()
                f_data_by_line = f_data.split("\n")
                num_ent += len(f_data_by_line)
                corpus_data.append(f_data_by_line)
    
    print(f"\nLoaded {num_ent:,} text entries from {len(class_tokens)-1} categories")
    return corpus_data, class_tokens


def analyze_token_frequencies(tokenizer: Tokenizer, corpus_data: List[List[str]]) -> Dict[int, int]:
    """
    Analyze token frequencies in the corpus.
    
    Args:
        tokenizer: Trained tokenizer
        corpus_data: List of lists, where each inner list contains entries from one file
    
    Returns:
        Dictionary mapping token_id to frequency count
    """
    print("\nEncoding corpus and counting token frequencies...")
    
    freq = {}
    
    voc_len = tokenizer.get_vocab_size()
    
    
    for token_id in range(voc_len):
        freq[token_id] = 0
        
    total_entries = sum(len(file_entries) for file_entries in corpus_data)
    # Iterate over each file's entries
    with tqdm(total=total_entries, desc="Processing entries") as pbar:
        for file_entries in corpus_data:
            for entry in file_entries:
                if entry:  # Make sure entry is not empty
                    # Encode the text entry
                    encoded = tokenizer.encode(entry)
                    # Count each token in the encoded sequence
                    for token_id in encoded.ids:
                        freq[token_id] += 1
                pbar.update(1)
    
    return freq


def categorize_frequencies(frequencies: Dict[int, int]) -> Dict[str, int]:
    """
    Categorize tokens by frequency ranges.
    
    Returns:
        Dictionary with category names and counts
    """
    categories = {
        '<100': 0,
        '100-999': 0,
        '1,000-4,999': 0,
        '5,000-9,999': 0,
        '10,000-24,999': 0,
        '25,000-49,999': 0,
        '50,000-99,999': 0,
        '100,000-499,999': 0,
        '500,000-999,999': 0,
        '1,000,000-4,999,999': 0,
        '5,000,000-9,999,999': 0,
        '10,000,000-14,999,999': 0,
        '15,000,000-19,999,999': 0,
        '20,000,000-29,999,999': 0,
        '>=30,000,000': 0
    }
    
    for freq in frequencies.values():
        if freq < 100:
            categories['<100'] += 1
        elif freq < 1000:
            categories['100-999'] += 1
        elif freq < 5000:
            categories['1,000-4,999'] += 1
        elif freq < 10000:
            categories['5,000-9,999'] += 1
        elif freq < 25000:
            categories['10,000-24,999'] += 1
        elif freq < 50000:
            categories['25,000-49,999'] += 1
        elif freq < 100000:
            categories['50,000-99,999'] += 1
        elif freq < 500000:
            categories['100,000-499,999'] += 1
        elif freq < 1000000:
            categories['500,000-999,999'] += 1
        elif freq < 5000000:
            categories['1,000,000-4,999,999'] += 1
        elif freq < 10000000:
            categories['5,000,000-9,999,999'] += 1
        elif freq < 15000000:
            categories['10,000,000-14,999,999'] += 1
        elif freq < 20000000:
            categories['15,000,000-19,999,999'] += 1
        elif freq < 30000000:
            categories['20,000,000-29,999,999'] += 1
        else:
            categories['>=30,000,000'] += 1
    
    return categories


def get_top_tokens(tokenizer: Tokenizer, frequencies: Dict[int, int], n: int = 20) -> List[Tuple[str, int]]:
    """Get the top N most frequent tokens."""
    sorted_tokens = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:n]
    
    top_tokens = []
    for token_id, freq in sorted_tokens:
        token_str = tokenizer.decode([token_id])
        top_tokens.append((token_str, freq))
    
    return top_tokens


def get_least_frequent_tokens(tokenizer: Tokenizer, frequencies: Dict[int, int], n: int = 20) -> List[Tuple[str, int]]:
    """Get the least frequent tokens that actually appear."""
    # Filter out tokens with zero frequency (if any)
    non_zero = [(tid, freq) for tid, freq in frequencies.items() if freq > 0]
    sorted_tokens = sorted(non_zero, key=lambda x: x[1])[:n]
    
    least_tokens = []
    for token_id, freq in sorted_tokens:
        token_str = tokenizer.decode([token_id])
        least_tokens.append((token_str, freq))
    
    return least_tokens


def print_frequency_report(categories: Dict[str, int], total_tokens: int, vocab_size: int):
    """Print formatted frequency report."""
    print("\n" + "="*60)
    print("TOKEN FREQUENCY DISTRIBUTION REPORT")
    print("="*60)
    print(f"\nTotal vocabulary size: {vocab_size:,}")
    print(f"Unique tokens found in corpus: {total_tokens:,}")
    print(f"Tokens not found in corpus: {vocab_size - total_tokens:,}")
    
    print("\n" + "-"*60)
    print("FREQUENCY CATEGORIES:")
    print("-"*60)
    
    for category, count in categories.items():
        percentage = (count / vocab_size) * 100
        print(f"  {category:20s} : {count:8,} tokens ({percentage:5.2f}%)")
    
    print("-"*60)
    print(f"  {'TOTAL':20s} : {vocab_size:8,} tokens (100.00%)")
    print("="*60)


def plot_frequency_distribution(categories: Dict[str, int], save_path: str = None):
    """Create a bar chart of frequency distribution."""
    # Define order for categories
    category_order = [
        '<100',
        '100-999',
        '1,000-4,999',
        '5,000-9,999',
        '10,000-24,999',
        '25,000-49,999',
        '50,000-99,999',
        '100,000-499,999',
        '500,000-999,999',
        '1,000,000-4,999,999',
        '5,000,000-9,999,999',
        '10,000,000-14,999,999',
        '15,000,000-19,999,999',
        '20,000,000-29,999,999',
        '>=30,000,000'
    ]
    
    # Get values in order
    ordered_labels = [cat for cat in category_order if cat in categories]
    ordered_counts = [categories[cat] for cat in ordered_labels]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(ordered_labels)), ordered_counts, color='steelblue', edgecolor='black')
    plt.xlabel('Frequency Range')
    plt.ylabel('Number of Tokens')
    plt.title('Token Frequency Distribution in Corpus')
    plt.xticks(range(len(ordered_labels)), ordered_labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, ordered_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("TOKEN FREQUENCY ANALYZER")
    print("="*60)
        
    # Step 2: Get vocabulary size
    print("\n2. CONFIGURATION")
    print("-"*40)
    
    try:
        vocab_size = int(input("Enter vocabulary size for tokenizer training: "))
    except ValueError:
        print("Invalid input. Using default: 8192")
        vocab_size = 8192
    
    # Step 3: Load and process corpus
    print("\n3. LOADING CORPUS")
    print("-"*40)
    
    try:
        corpus_texts, class_tokens = load_corpus_data('data')
        
        if not corpus_texts:
            print("No text data found in 'data' directory.")
            print("Please ensure your data is organized in folders under 'data'.")
            return
        
        # Step 4: Train tokenizer
        print("\n4. TRAINING TOKENIZER")
        print("-"*40)
        
        print(f"Training tokenizer with vocabulary size: {vocab_size:,}")
        tokenizer = train_tokenizer(
            vocab_size,
            SPECIAL_TOKENS,
            UNKNOWN_TOKEN,
            class_tokens,
            corpus_texts,
            train=True
        )
        
        # Step 5: Analyze frequencies
        print("\n5. ANALYZING TOKEN FREQUENCIES")
        print("-"*40)
        
        frequencies = analyze_token_frequencies(tokenizer, corpus_texts)
        for token_id in frequencies:
            print(f"{token_id} [{tokenizer.decode([token_id])}]: {frequencies.get(token_id)}")
        
        
        
        
        
        
        # Step 6: Categorize and report
        print("\n6. GENERATING REPORT")
        print("-"*40)
        
        categories = categorize_frequencies(frequencies)
        print_frequency_report(categories, len(frequencies), vocab_size)
        
        # Step 7: Show top and least frequent tokens
        print("\n" + "="*60)
        print("TOP 20 MOST FREQUENT TOKENS:")
        print("="*60)
        top_tokens = get_top_tokens(tokenizer, frequencies)
        for i, (token, freq) in enumerate(top_tokens, 1):
            # Handle special tokens for display
            display_token = token.replace('\n', '\\n').replace('\t', '\\t')
            if len(display_token) > 40:
                display_token = display_token[:37] + "..."
            print(f"  {i:2d}. '{display_token:40s}' : {freq:10,}")
        
        print("\n" + "="*60)
        print("20 LEAST FREQUENT TOKENS (that appear):")
        print("="*60)
        least_tokens = get_least_frequent_tokens(tokenizer, frequencies)
        for i, (token, freq) in enumerate(least_tokens, 1):
            display_token = token.replace('\n', '\\n').replace('\t', '\\t')
            if len(display_token) > 40:
                display_token = display_token[:37] + "..."
            print(f"  {i:2d}. '{display_token:40s}' : {freq:10,}")
        
        # Step 8: Create visualization
        print("\n7. CREATING VISUALIZATION")
        print("-"*40)
        
        plot_choice = prompt_select(
            "Would you like to save a visualization?",
            [
                Choice(title="Yes, save as PNG", value="save"),
                Choice(title="Show in window", value="show"),
                Choice(title="Skip", value="skip")
            ]
        )
        
        if plot_choice == "save":
            save_path = f"token_frequency_{folder_choice}.png"
            plot_frequency_distribution(categories, save_path)
        elif plot_choice == "show":
            plot_frequency_distribution(categories)
        
        # Step 9: Save frequency data
        print("\n8. SAVING RESULTS")
        print("-"*40)
        
        save_choice = prompt_select(
            "Save frequency data to file?",
            [
                Choice(title="Yes", value=True),
                Choice(title="No", value=False)
            ]
        )
        
        if save_choice:
            output_path = f"token_frequencies_{folder_choice}.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'frequencies': frequencies,
                    'categories': categories,
                    'vocab_size': vocab_size,
                    'total_tokens_found': len(frequencies)
                }, f)
            print(f"Frequency data saved to: {output_path}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please make sure your data is organized in folders under 'data/'")
        print("Each folder should contain text files with names like 'category-names.txt'")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()