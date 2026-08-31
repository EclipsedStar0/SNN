#!/usr/bin/env python3
"""
String Deduplication Script
Removes highly similar sections from an array of strings while preserving order.
Uses rolling hash (Rabin-Karp) for efficient similarity detection.

Memory-optimized version: does NOT store chunk text in memory.
"""

from tqdm.auto import tqdm
from typing import Dict, Set, List
from pathlib import Path
import ftfy
import time


class SimilarityDeduplicator:
    """Memory-efficient deduplicator using rolling hash windows."""
    
    def __init__(self, window_size: int = 100, similarity_threshold: float = 0.8):
        """
        Initialize the deduplicator.
        
        Args:
            window_size: Size of the sliding window for comparison (characters)
            similarity_threshold: Fraction of matching windows to consider sections similar (0.0-1.0)
        """
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[int] = set()
        self.stats = {
            'chunks_seen': 0,
            'chunks_kept': 0,
            'chunks_removed': 0
        }
    
    def _get_rolling_hashes(self, text: str) -> List[int]:
        """
        Generate rolling hashes for a text using a sliding window.
        MEMORY EFFICIENT: Only stores hashes, not text.
        
        Args:
            text: Input string
            
        Returns:
            List of hash values (integers only)
        """
        if len(text) < self.window_size:
            return [hash(text)]
        
        hashes = []
        for i in range(len(text) - self.window_size + 1):
            window = text[i:i + self.window_size]
            hashes.append(hash(window))
        
        return hashes
    
    def deduplicate(self, catalog: Dict[str, List]) -> Dict[str, List]:
        """
        Deduplicate text entries by removing similar chunks.
        
        Args:
            catalog: Dict mapping entry_id -> [filename, subfolder, text]
            
        Returns:
            New catalog with deduplicated text
        """
        new_catalog = {}
        self.seen_hashes = set()
        
        for entry_id in tqdm(catalog, desc="Deduplicating"):
            filename, subfolder, text = catalog[entry_id]
            
            # Get rolling hashes (memory efficient - integers only)
            hash_list = self._get_rolling_hashes(text)
            
            # Track which positions to keep
            keep_positions = []
            
            for i, chunk_hash in enumerate(hash_list):
                self.stats['chunks_seen'] += 1
                
                if chunk_hash not in self.seen_hashes:
                    keep_positions.append(i)
                    self.seen_hashes.add(chunk_hash)
                    self.stats['chunks_kept'] += 1
                else:
                    self.stats['chunks_removed'] += 1
            
            # Build deduplicated text from keep_positions
            if not keep_positions:
                # Everything was a duplicate - keep a small portion
                deduplicated_text = text[:self.window_size] if len(text) >= self.window_size else text
            else:
                # Merge kept windows efficiently
                deduplicated_parts = []
                last_end = 0
                
                for pos in keep_positions:
                    start = pos
                    end = pos + self.window_size
                    
                    # Only add non-overlapping parts
                    if start >= last_end:
                        deduplicated_parts.append(text[start:end])
                        last_end = end
                    elif end > last_end:
                        # Partial overlap - add only new part
                        deduplicated_parts.append(text[last_end:end])
                        last_end = end
                
                # Add any remaining text
                if last_end < len(text):
                    remaining = text[last_end:]
                    if len(remaining) > self.window_size // 10:  # Only if substantial
                        deduplicated_parts.append(remaining)
                
                deduplicated_text = ''.join(deduplicated_parts)
            
            new_catalog[entry_id] = [filename, subfolder, deduplicated_text]
        
        return new_catalog
    
    def get_stats(self) -> Dict[str, int]:
        """Return deduplication statistics."""
        return self.stats.copy()


class EfficientSimilarityDeduplicator:
    """
    Entry-level deduplicator - checks if entire documents are similar.
    MUCH more memory efficient for large corpora.
    """
    
    def __init__(self, window_size: int = 100, similarity_threshold: float = 0.8):
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[int] = set()
        self.stats = {
            'entries_seen': 0,
            'entries_kept': 0,
            'entries_removed': 0,
            'chunks_seen': 0,
            'chunks_kept': 0,
            'chunks_removed': 0
        }
    
    def deduplicate(self, catalog: Dict[str, List]) -> Dict[str, List]:
        """
        Deduplicate by checking if entries are similar to previously seen content.
        Much faster and more memory efficient than chunk-level deduplication.
        """
        new_catalog = {}
        self.seen_hashes = set()
        
        for entry_id in tqdm(catalog, desc="Deduplicating"):
            filename, subfolder, text = catalog[entry_id]
            self.stats['entries_seen'] += 1
            
            # Generate hashes for this entry
            if len(text) < self.window_size:
                current_hashes = [hash(text)]
            else:
                current_hashes = []
                # Sample every Nth window to reduce memory (stride approach)
                stride = max(1, self.window_size // 4)  # Sample every 1/4 window
                for i in range(0, len(text) - self.window_size + 1, stride):
                    window = text[i:i + self.window_size]
                    current_hashes.append(hash(window))
            
            self.stats['chunks_seen'] += len(current_hashes)
            
            # Check overlap with seen content
            if not current_hashes:
                new_catalog[entry_id] = [filename, subfolder, text]
                self.stats['entries_kept'] += 1
                continue
            
            seen_count = sum(1 for h in current_hashes if h in self.seen_hashes)
            overlap_ratio = seen_count / len(current_hashes)
            
            if overlap_ratio < self.similarity_threshold:
                # Keep this entry
                new_catalog[entry_id] = [filename, subfolder, text]
                self.seen_hashes.update(current_hashes)
                self.stats['entries_kept'] += 1
                self.stats['chunks_kept'] += len(current_hashes)
            else:
                # Skip - too similar
                self.stats['entries_removed'] += 1
                self.stats['chunks_removed'] += len(current_hashes)
        
        return new_catalog
    
    def get_stats(self) -> Dict[str, int]:
        """Return deduplication statistics."""
        return self.stats.copy()


def deduplicate_strings(
    catalog: Dict[str, List],
    window_size: int = 100,
    similarity_threshold: float = 0.8,
    mode: str = 'efficient'
) -> Dict[str, List]:
    """
    Deduplicate strings in catalog.
    
    Args:
        catalog: Dict mapping entry_id -> [filename, subfolder, text]
        window_size: Size of rolling hash window
        similarity_threshold: Threshold for considering content similar
        mode: 'efficient' (entry-level, recommended) or 'precise' (chunk-level)
        
    Returns:
        Deduplicated catalog
    """
    if mode == 'precise':
        deduplicator = SimilarityDeduplicator(window_size, similarity_threshold)
    else:
        deduplicator = EfficientSimilarityDeduplicator(window_size, similarity_threshold)
    
    result = deduplicator.deduplicate(catalog)
    
    # Print statistics
    stats = deduplicator.get_stats()
    print(f"\nDeduplication Statistics:")
    for key, value in stats.items():
        if 'seen' in key and value > 0:
            print(f"  {key.replace('_', ' ').title()}: {value:,}")
    
    if 'chunks_kept' in stats and stats['chunks_seen'] > 0:
        kept_pct = 100 * stats['chunks_kept'] / stats['chunks_seen']
        removed_pct = 100 * stats['chunks_removed'] / stats['chunks_seen']
        print(f"  Kept: {kept_pct:.2f}% | Removed: {removed_pct:.2f}%")
    
    return result


def get_file_names(directory_path: str, sort: bool = True, exclude_hidden: bool = False) -> List[str]:
    """Get file names from a directory."""
    try:
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory '{directory_path}' not found.")
        if not path.is_dir():
            raise NotADirectoryError(f"'{directory_path}' is not a directory.")
        
        files = [
            item.name for item in path.iterdir()
            if item.is_file() and (not exclude_hidden or not item.name.startswith('.'))
        ]
        
        return sorted(files) if sort else files
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def get_folder_names(directory_path: str, sort: bool = True, exclude_hidden: bool = False) -> List[str]:
    """Get folder names from a directory."""
    try:
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory '{directory_path}' not found.")
        if not path.is_dir():
            raise NotADirectoryError(f"'{directory_path}' is not a directory.")
        
        folders = [
            item.name for item in path.iterdir()
            if item.is_dir() and (not exclude_hidden or not item.name.startswith('.'))
        ]
        
        return sorted(folders) if sort else folders
        
    except Exception as e:
        print(f"Error: {e}")
        return []

import re
def normalize_newlines(content):
    """
    Convert multiple consecutive newlines according to the rules:
    1 newline -> 0 newlines
    2 newlines -> 1 newline
    3 newlines -> 2 newlines
    4 newlines -> 3 newlines
    5 newlines -> 4 newlines
    More than 5 newlines -> 5 newlines (or 4 based on your spec)
    """
    # IMPORTANT: Process from smallest to largest to avoid cascading replacements
    # First replace single newlines with nothing
    content = re.sub(r'(?<!\n)\n(?!\n)', ' ', content)
    
    # Then replace double newlines with single newline
    content = content.replace('\n' * 2, '\n' * 1)
    
    # Then replace triple newlines with double newlines
    content = content.replace('\n' * 3, '\n' * 2)
    
    # Then replace quadruple newlines with triple newlines
    content = content.replace('\n' * 4, '\n' * 3)
    
    # Then replace quintuple newlines with quadruple newlines
    content = content.replace('\n' * 5, '\n' * 3)
    
    # Finally, for more than 5 newlines, replace with 4 newlines
    # (or 5 if you want to keep the pattern consistent)
    content = re.sub(r'\n{6,}', '\n' * 3, content)
    
    return content

def normalize_white_space_regex(content):
    # Match 1 or more consecutive newlines
    return re.sub(r' +', ' ', content)
    
def normalize_tab_space_regex(content):
    return re.sub(r'\t ', '\t', content)
    
def normalize_tab_space_regex_p2(content):
    return re.sub(r' \t', '\t', content)


def load_corpus(folder_to_search: str) -> tuple:
    """
    Load all text files from a folder structure.
    
    Returns:
        catalog, words_per_folder, total_words, total_characters, total_works
    """
    all_sub_folders = get_folder_names(folder_to_search)
    
    
    total_characters = 0
    total_words = 0
    total_works = 0
    
    new_total_words = 0
    new_total_characters = 0
    
    words_for_folders = {}
    new_words_for_folders = {}
    
    top_f_names = {}
    catalog = {}
    
    for sub_folder_name in tqdm(all_sub_folders, desc='Scanning folders'):
        words_for_folders[sub_folder_name] = 0
        new_words_for_folders[sub_folder_name] = 0
        
        files_in_folder = get_file_names(f'{folder_to_search}/{sub_folder_name}')
        
        for file_name in tqdm(files_in_folder, desc=f'  {sub_folder_name}', leave=False):
            if file_name.endswith('.txt'):
                total_works += 1
                file_path = Path(folder_to_search) / sub_folder_name / file_name
                
                with open(file_path, 'r', encoding="utf-8") as file:
                    text = file.read()
                    text = ftfy.fix_text(text)
                    
                    num_characters = len(text)
                    num_words = len(text.split())
                    words_for_folders[sub_folder_name] += num_words
                    total_words += num_words
                    total_characters += num_characters
                    
                    text = normalize_white_space_regex(text)
                    text = normalize_tab_space_regex(text)
                    text = normalize_tab_space_regex_p2(text)
                    
                    tripped_any_license = False
                    t_text = text.split("***** This file should be named")
                    if len(t_text) > 1:
                        text = t_text[0]
                    
                    t_text = text.split("START: FULL LICENSE\nTHE FULL PROJECT GUTENBERG LICENSE\nPLEASE READ THIS BEFORE YOU DISTRIBUTE OR USE THIS WORK")
                    if len(t_text) > 1:
                        tripped_any_license = True
                        text = t_text[0]
                    else: 
                        t_text = text.split("START: FULL LICENSE\n\nTHE FULL PROJECT GUTENBERG LICENSE\n\nPLEASE READ THIS BEFORE YOU DISTRIBUTE OR USE THIS WORK")
                        if len(t_text) > 1:
                            tripped_any_license = True
                            text = normalize_newlines(t_text[0])
                        else:
                            t_text = text.split("START: FULL LICENSE\n\n\nTHE FULL PROJECT GUTENBERG LICENSE\n\n\nPLEASE READ THIS BEFORE YOU DISTRIBUTE OR USE THIS WORK")
                            if len(t_text) > 1:
                                tripped_any_license = True
                                text = normalize_newlines(t_text[0])
                                text = normalize_newlines(text)
                            else:
                                t_text = text.split("START: FULL LICENSE\n\n\n\nTHE FULL PROJECT GUTENBERG LICENSE\n\n\n\nPLEASE READ THIS BEFORE YOU DISTRIBUTE OR USE THIS WORK")
                                if len(t_text) > 1:
                                    tripped_any_license = True
                                    text = normalize_newlines(t_text[0])
                                    text = normalize_newlines(text)
                                    text = normalize_newlines(text)
                                
                    t_text = text.split("If you are not located in the United States, you will have to check the laws of the country where you are located before using this eBook.")
                    if len(t_text) > 1:
                        tripped_any_license = True
                        text = t_text[1]
                    else:
                        t_text = text.split("If you are not located in the United States,\nyou will have to check the laws of the country where you are located\nbefore using this eBook.")
                        if len(t_text) > 1:
                            tripped_any_license = True
                            text = normalize_newlines(t_text[1])
                        else:                    
                            t_text = text.split("If you are not located in the United States,\n\nyou will have to check the laws of the country where you are located\n\nbefore using this eBook.")
                            if len(t_text) > 1:
                                tripped_any_license = True
                                text = normalize_newlines(t_text[1])
                                text = normalize_newlines(text)
                            else:                    
                                t_text = text.split("If you are not located in the United States,\n\n\nyou will have to check the laws of the country where you are located\n\nbefore using this eBook.")
                                if len(t_text) > 1:
                                    tripped_any_license = True
                                    text = normalize_newlines(t_text[1])
                                    text = normalize_newlines(text)
                                    text = normalize_newlines(text)
                                else:      
                                    t_text = text.split("You may copy it, give it away or re-use it under the terms of the Project Gutenberg License included with this eBook or online at www.gutenberg.org")
                                    if len(t_text) > 1:
                                        tripped_any_license = True
                                        text = t_text[1]
                                    else:
                                        t_text = text.split("You may copy it, give it away or\nre-use it under the terms of the Project Gutenberg License included\nwith this eBook or online at www.gutenberg.org")
                                        if len(t_text) > 1:
                                            tripped_any_license = True
                                            text = normalize_newlines(t_text[1])
                                        else:                    
                                            t_text = text.split("You may copy it, give it away or\n\nre-use it under the terms of the Project Gutenberg License included\n\nwith this eBook or online at www.gutenberg.org")
                                            if len(t_text) > 1:
                                                tripped_any_license = True
                                                text = normalize_newlines(t_text[1])
                                                text = normalize_newlines(text)
                                            else:
                                                t_text = text.split("You may copy it, give it away or\n\n\nre-use it under the terms of the Project Gutenberg License included\n\n\nwith this eBook or online at www.gutenberg.org")
                                                if len(t_text) > 1:
                                                    tripped_any_license = True
                                                    text = normalize_newlines(t_text[1])
                                                    text = normalize_newlines(text)
                                                    text = normalize_newlines(text)
                                                    
                    t_text = text.split("Updated editions will replace the previous one—the old editions will be renamed.\n\nCreating the works from print editions not protected by U.S. copyright law means that no one owns a United States copyright in these works")
                    if len(t_text) > 1:
                        tripped_any_license = True
                        text = normalize_newlines(t_text[0])
                    else:
                        t_text = text.split("Updated editions will replace the previous one—the old editions will be renamed.\n\n\nCreating the works from print editions not protected by U.S. copyright law means that no one owns a United States copyright in these works")
                        if len(t_text) > 1:
                            tripped_any_license = True
                            text = normalize_newlines(t_text[0])
                            text = normalize_newlines(text)
                    
                    if not tripped_any_license:
                        text = normalize_newlines(text)
                        text = normalize_newlines(text)
                    
                    new_words = len(text.split())
                    new_characters = len(text)
                    new_words_for_folders[sub_folder_name] += new_words
                    new_total_words += new_words
                    new_total_characters += new_characters
                    
                    # Use unique key
                    entry_key = f"{sub_folder_name}/{file_name}"
                    catalog[entry_key] = [file_name, sub_folder_name, text]
                    
        t_name = sub_folder_name.split("- ")
        if len(t_name) > 1:
            if t_name[0] not in top_f_names:
                top_f_names[t_name[0]] = {
                    'Categories': [],
                    'BWCount': 0,
                    'AWCount': 0,
                }
            top_f_names[t_name[0]]['Categories'].append(sub_folder_name)
            top_f_names[t_name[0]]['BWCount'] += words_for_folders.get(sub_folder_name)
            top_f_names[t_name[0]]['AWCount'] += new_words_for_folders.get(sub_folder_name)
        else:
            top_f_names[sub_folder_name] = {
                    'Categories': [sub_folder_name],
                    'BWCount': words_for_folders.get(sub_folder_name),
                    'AWCount': new_words_for_folders.get(sub_folder_name),
                }
                    
                    
    
    return catalog, top_f_names, words_for_folders, new_words_for_folders, total_words, new_total_words, total_characters, new_total_characters, total_works


def save_corpus(catalog: Dict[str, List], output_folder: str):
    """Save deduplicated corpus to disk."""
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Create all necessary subfolders
    subfolders = set(entry[1] for entry in catalog.values())
    for subfolder in subfolders:
        (output_path / subfolder).mkdir(exist_ok=True)
    
    # Save files
    for entry_key, (filename, subfolder, text) in tqdm(catalog.items(), desc="Saving files"):
        file_path = output_path / subfolder / filename
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)


# Example usage
if __name__ == "__main__":
    
    folder_to_search = "normalised_files"
    deduplicated_folder = 'deduplicated'
    trim_folder = 'trimmed_liscensing'
    
    # Load corpus
    print("Loading corpus...")
    catalog, top_f_names, words_for_folders, new_folder_words, total_words, new_t_words, total_characters, new_t_charas, total_works = load_corpus(folder_to_search)
    
    print(f"\nCorpus Statistics:")
    print(f"  Total Books: {total_works:,}")
    print(f"  Total Words: {total_words:,}")
    print(f"  Total Characters: {total_characters:,}")
    
    for top_folder in top_f_names:
        if top_f_names[top_folder]['BWCount'] == 0:
            continue
        print(f"{top_folder}: {top_f_names[top_folder]['BWCount']:,}, ({100*top_f_names[top_folder]['BWCount']/total_words:.2f}%)")
        for sub_folder in top_f_names[top_folder]['Categories']:
            if words_for_folders[sub_folder] == 0:
                continue
            
            print(f"\t{sub_folder}: {words_for_folders[sub_folder]:,}, ({100*words_for_folders[sub_folder]/top_f_names[top_folder]['BWCount']:.2f}%)")
    
    
    
    print(f"\nCorpus Statistics (After Liscense Removal):")
    print(f"  Total Books: {total_works:,}")
    print(f"  Total Words: {new_t_words:,}")
    print(f"  Total Characters: {new_t_charas:,}")
    
    for top_folder in top_f_names:
        if top_f_names[top_folder]['AWCount'] == 0:
            continue
        print(f"{top_folder}: {top_f_names[top_folder]['AWCount']:,}, ({100*top_f_names[top_folder]['AWCount']/new_t_words:.2f}%)")
        for sub_folder in top_f_names[top_folder]['Categories']:
            if new_folder_words[sub_folder] == 0:
                continue
            
            print(f"\t{sub_folder}: {new_folder_words[sub_folder]:,}, ({100*new_folder_words[sub_folder]/top_f_names[top_folder]['AWCount']:.2f}%)")

    print(f"\nSaving trimmed corpus to '{trim_folder}'...")
    save_corpus(catalog, trim_folder)
    
    raise Exception("don't pass")
    
    threshold_to_use = 0.70
    mode_to_use = 'precise'
    
    # Deduplicate
    print(f"\nStarting deduplication...")
    print(f"  Window size: 5000 characters")
    print(f"  Similarity threshold: {threshold_to_use}")
    print(f"  Mode: {mode_to_use}")
    
    start = time.time()
    new_catalog = deduplicate_strings(
        catalog, 
        window_size=5000, 
        similarity_threshold={threshold_to_use},
        mode=mode_to_use  # Use 'precise' for chunk-level (more memory!)
    )
    elapsed = time.time() - start
    
    print(f"\nDeduplication completed in {elapsed:.2f} seconds")
    catalog = {}
    
    # Save deduplicated corpus
    print(f"\nSaving deduplicated corpus to '{deduplicated_folder}'...")
    save_corpus(new_catalog, deduplicated_folder)
    
    # Calculate statistics
    new_total_words = 0
    for entry in tqdm(new_catalog, desc='Tallying up new word count...'):
        new_total_words += len(new_catalog[entry][2].split())
        
    reduction_factor = new_total_words / num_words
    
    print(f"\nFinal Statistics:")
    print(f"  Original words: {num_words:,}")
    print(f"  Deduplicated words: {new_total_words:,}")
    print(f"  Reduction factor: {reduction_factor:.4f}")
    print(f"  Removed: {100*(1-reduction_factor):.2f}%")
    
    print("\nDone!")