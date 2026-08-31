import os
import pickle
import questionary
from questionary import Choice, Style
from typing import Any, TypeVar


from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import re
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from tqdm.auto import tqdm
import ftfy
from collections import defaultdict
import datasets
import math


SEP_TOKEN = "<SEP>"
UNKNOWN_TOKEN = "<UNK>"
PADDING_TOKEN = "<PAD>"
BEGIN_OF_STREAM_TOKEN = "<BOS>"
END_OF_STREAM_TOKEN = "<EOS>"
RESPONSE_TOKEN = "<RESP>"
SPECIAL_TOKENS = [PADDING_TOKEN, BEGIN_OF_STREAM_TOKEN, END_OF_STREAM_TOKEN, UNKNOWN_TOKEN]


def train_tokenizer(vocab_size, special_tokens_param, unk_tok, added_toks, training_data, train=True):
        """Train BPE tokenizer."""
        spec_plus_add = special_tokens_param + added_toks
        bpe_trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=spec_plus_add)
        tokenize_func = Tokenizer(models.BPE(unk_token=unk_tok))
        # tokenize_func.add_tokens(added_toks)
        tokenize_func.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenize_func.decoder = decoders.ByteLevel()
        
        if train:
            tokenize_func.train_from_iterator(training_data, trainer=bpe_trainer)
        
        return tokenize_func

def prompt_select(message: str, choices: list[Any]) -> Any:
    return questionary.select(
            message,
            choices=choices,
            style=Style([("highlighted", "reverse")]),
        ).ask()

def get_file_names(directory_path, sort=True, exclude_hidden=False):
    """
    Get file names from a directory with options.
    
    Args:
        directory_path (str): Path to the directory
        sort (bool): Whether to sort file names alphabetically
        exclude_hidden (bool): Whether to exclude hidden files (starting with .)
    
    Returns:
        list: List of file names
    """
    try:
        # Convert to Path object for better handling
        path = Path(directory_path)
        
        # Check if path exists
        if not path.exists():
            raise FileNotFoundError(f"Directory '{directory_path}' not found.")
        
        # Check if it's a directory
        if not path.is_dir():
            raise NotADirectoryError(f"'{directory_path}' is not a directory.")
        
        # Get all files
        files = []
        for item in path.iterdir():
            if item.is_file():
                file_name = item.name
                # Skip hidden files if requested
                if exclude_hidden and file_name.startswith('.'):
                    continue
                files.append(file_name)
        
        # Sort if requested
        if sort:
            files.sort()
        
        return files
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def get_folder_names(directory_path, sort=True, exclude_hidden=False):
    """
    Get folder names from a directory with options.
    
    Args:
        directory_path (str): Path to the directory
        sort (bool): Whether to sort folder names alphabetically
        exclude_hidden (bool): Whether to exclude hidden folders (starting with .)
    
    Returns:
        list: List of folder names
    """
    try:
        # Convert to Path object for better handling
        path = Path(directory_path)
        
        # Check if path exists
        if not path.exists():
            raise FileNotFoundError(f"Directory '{directory_path}' not found.")
        
        # Check if it's a directory
        if not path.is_dir():
            raise NotADirectoryError(f"'{directory_path}' is not a directory.")
        
        # Get all directories
        folders = []
        for item in path.iterdir():
            if item.is_dir():
                folder_name = item.name
                # Skip hidden folders if requested
                if exclude_hidden and folder_name.startswith('.'):
                    continue
                folders.append(folder_name)
        
        # Sort if requested
        if sort:
            folders.sort()
        
        return folders
        
    except Exception as e:
        print(f"Error: {e}")
        return []


folder_names = get_folder_names('data')
class_tokens = [RESPONSE_TOKEN]

corpus_data = []
for folder in tqdm(folder_names, desc='Processing folders...'):
    files_in_folder = get_file_names(f'data/{folder}')
    # class_tokens.append(f'<{folder}>')
    for file in tqdm(files_in_folder, desc=f'Processing {folder}...'):
        ac_identifier = file.split("-names.txt")[0]
        class_tokens.append(f'<{ac_identifier}>')
        with open(f'data/{folder}/{file}', 'r', encoding='utf-8') as r_file:
            f_data = r_file.read()
            f_data_by_line = f_data.split("\n")
            
            corpus_data.append(f_data_by_line)

max_vocab_len = int(input('Vocab Size: '))
TOKENIZER = train_tokenizer(max_vocab_len, SPECIAL_TOKENS, UNKNOWN_TOKEN, class_tokens, corpus_data)

freq_dict = {}
voc_len = TOKENIZER.get_vocab_size()
for index in range(voc_len):
    freq_dict[index] = 0
    
for file_arr in tqdm(corpus_data, desc='Processing files...'):
    for entry in tqdm(file_arr, desc='Processing file...'):
        encoded_text = TOKENIZER.encode(entry).ids
        for token_id in encoded_text:
            freq_dict[token_id] += 1
            
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

for freq in freq_dict.values():
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

print(categories)
            
            
            

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
            
print("\n" + "="*60)
print("TOP 20 MOST FREQUENT TOKENS:")
print("="*60)
top_tokens = get_top_tokens(TOKENIZER, freq_dict)
for i, (token, freq) in enumerate(top_tokens, 1):
    # Handle special tokens for display
    display_token = token.replace('\n', '\\n').replace('\t', '\\t')
    if len(display_token) > 40:
        display_token = display_token[:37] + "..."
    print(f"  {i:2d}. '{display_token:40s}' : {freq:10,}")

print("\n" + "="*60)
print("20 LEAST FREQUENT TOKENS (that appear):")
print("="*60)
least_tokens = get_least_frequent_tokens(TOKENIZER, freq_dict)
for i, (token, freq) in enumerate(least_tokens, 1):
    display_token = token.replace('\n', '\\n').replace('\t', '\\t')
    if len(display_token) > 40:
        display_token = display_token[:37] + "..."
    print(f"  {i:2d}. '{display_token:40s}' : {freq:10,}")
        
    

