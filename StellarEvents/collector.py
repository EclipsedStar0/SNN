import os
import pickle
import questionary
from questionary import Choice, Style
from typing import Any, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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


def train_tokenizer(vocab_size_param, special_tokens_param, unk_tok, added_toks, training_data, train=True):
        print(f'VOC SIZE PROVIDED: {vocab_size_param}')
        #print(f'SPEC Provided: {special_tokens_param}')
        #print(f'UNK Provided: {unk_tok}')
        #print(f'Added Tokens: {added_toks}')
        # print(f'First 1000 Toks: {training_data[:1000]}')
        print(f'Len of Training: {len(training_data)}')
    
    
        """Train BPE tokenizer."""
        spec_tok = special_tokens_param + added_toks
        bpe_trainer = trainers.BpeTrainer(vocab_size=vocab_size_param, special_tokens=spec_tok)
        tokenizer = Tokenizer(models.BPE(unk_token=unk_tok))
        #tokenizer.add_tokens(added_toks)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()
        
        if train:
            tokenizer.train_from_iterator(training_data, trainer=bpe_trainer)
        
        return tokenizer

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
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        least_tokens.append((token_str, freq))
    
    return least_tokens
    
def get_voc_freq(tokenizer, corpus_data, pres_tokens_param):
    freq_dict = {}
    voc_len = tokenizer.get_vocab_size()
    for index in range(voc_len):
        freq_dict[index] = 0
    deco_pres = []
    for token in pres_tokens_param:
        deco_pres.append(tokenizer.token_to_id(token))
    
    for entry in tqdm(corpus_data, desc='Processing files...'):
        encoded_text = tokenizer.encode(entry).ids
        for token_id in encoded_text:
            freq_dict[token_id] += 1
    
        

    num_cal = 0
    for token_id in freq_dict:
        if token_id in deco_pres and freq_dict[token_id] < 100:
            #print(f'{tokenizer.id_to_token(token_id):40s} {freq_dict[token_id]:10,}')
            print(tokenizer.id_to_token(token_id))
            num_cal += 1
    print(f'NUM: {num_cal}') 
    print(f'FOR : {freq_dict[tokenizer.token_to_id(":")]}')
                
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
    sec_cat = {}
    for key in categories:
        sec_cat[key] = [categories[key], f'{100*categories[key]/voc_len:.2f}%']
    


    print(sec_cat)

    print("\n" + "="*60)
    print("TOP 10 MOST FREQUENT TOKENS:")
    print("="*60)
    top_tokens = get_top_tokens(tokenizer, freq_dict, 10)
    for i, (token, freq) in enumerate(top_tokens, 1):
        # Handle special tokens for display
        display_token = token.replace('\n', '\\n').replace('\t', '\\t')
        if len(display_token) > 40:
            display_token = display_token[:37] + "..."
        print(f"  {i:2d}. '{display_token:40s}' : {freq:10,}")

    print("\n" + "="*60)
    print("10 LEAST FREQUENT TOKENS (that appear):")
    print("="*60)
    least_tokens = get_least_frequent_tokens(tokenizer, freq_dict, 10)
    for i, (token, freq) in enumerate(least_tokens, 1):
        display_token = token.replace('\n', '\\n').replace('\t', '\\t')
        if len(display_token) > 40:
            display_token = display_token[:37] + "..."
        print(f"  {i:2d}. '{display_token:40s}' : {freq:10,}")
    
    
def try_diff_encodings(file_path):
    try:
        with open(file_path, 'r', encoding='ascii') as r_file:
            return r_file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='cp437') as r_file:
                return r_file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='utf-7') as r_file:
                    return r_file.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='utf-8') as r_file:
                        return r_file.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='utf-16') as r_file:
                            return r_file.read()
                    except UnicodeDecodeError:
                        try:
                            with open(file_path, 'r', encoding='utf-32') as r_file:
                                return r_file.read()
                        
                        except UnicodeDecodeError:
                            raise Exception
        
   
def load_preset_toks():
    no_use_list = []
    with open('tok_with_too_few_examples.txt', 'r', encoding='utf-8') as r_file:
        no_use_list = r_file.read().split('\n')
    
    filtered_list = ['count_', 'any_', 'every_', 'can_', 'has_', 'is_', 'num_', 'random_', 'ordered_', 'add_', 'set_', 'remove_']
    for entry in filtered_list:
        if entry in no_use_list:
            print(f'[WARNING]: [{entry}] is in the no use list!')
    
    with open('token_list.txt', 'r', encoding='utf-8') as r_file:
        toke_list = r_file.read().split('\n')
        
        for entry in toke_list:
            low_entry = entry.lower()
            if entry != '' and entry not in no_use_list and entry not in filtered_list and entry +'_' not in filtered_list:
                filtered_list.append(entry.lower())
        return filtered_list
    
def main():
    container_names = get_folder_names('unprocessed_events')
    event_arr = []
    for container in container_names:
        event_file_names = get_file_names(f'unprocessed_events/{container}/events')
        for event_file_name in event_file_names:
            event_arr.append(try_diff_encodings(f'unprocessed_events/{container}/events/{event_file_name}').lower())
            # print(len(event_arr[-1]))
    
    print(f'Num Evt Files: {len(event_arr)}')
    
    len_of_dat = len("\n".join(event_arr))
    print(f'Corpus Len: {len_of_dat}')
            
    VOCAB_SIZE = int(input('Vocab Size: '))
    pres_tokens = load_preset_toks()
    alpha_bet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i','j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'u', 'r', 's', 't', 'v' ,'w', 'x', 'y', 'z']
    numeral = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    symbols = ['<', '>', '=', '{', '}', '\n', '\t', '-', '#', ' ', '_', ':', '.', '"', "'"]
    pres_tokens.extend(alpha_bet)
    pres_tokens.extend(numeral)
    pres_tokens.extend(symbols)
    print(f'Added Token Length: {len(pres_tokens)}')
    
    TOKENIZER = train_tokenizer(VOCAB_SIZE, SPECIAL_TOKENS, UNKNOWN_TOKEN, load_preset_toks(), event_arr)
    # print(f'Vocab: {TOKENIZER.get_vocab()}')
    print(f'Vocab Len: {TOKENIZER.get_vocab_size()}')
    
    # raise Exception
    
    get_voc_freq(TOKENIZER, event_arr, pres_tokens)
    
    
    
    
    
    
    

if __name__ == "__main__":
    main()