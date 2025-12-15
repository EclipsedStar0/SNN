import re
import ftfy
from collections import defaultdict

class SimpleBPETokenizer:
    def __init__(self):
        self.vocab = []
        self.merges = {}
        self.vocab_size = 0
        
    def pre_tokenize(self, text):
        """Split text into words and handle spaces properly"""
        # Add special space markers between words
        text = re.sub(r'\s+', ' ', text.strip())
        words = text.split(' ')
        return words
    
    def get_stats(self, splits):
        """Get frequency of all adjacent pairs"""
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def merge_pair(self, a, b, splits):
        """Merge pair (a, b) in all splits"""
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
        return splits
    
    def train(self, texts, vocab_size=300):
        """Train BPE tokenizer on given texts"""
        # Step 1: Build word frequencies with space markers
        self.word_freqs = defaultdict(int)
        
        for text in texts:
            text = re.sub(r'\s+', ' ', text.strip())
            words = text.split(' ')
            for i, word in enumerate(words):
                if not word:
                    continue
                # Add space marker for all words except the first one
                if i > 0:
                    word_with_space = 'Ġ' + word
                else:
                    word_with_space = word
                self.word_freqs[word_with_space] += 1
        
        # Build initial alphabet
        alphabet = set()
        for word in self.word_freqs.keys():
            for char in word:
                alphabet.add(char)
        alphabet = sorted(alphabet)
        
        # Initial vocabulary: special tokens + alphabet
        self.vocab = ["<|endoftext|>"] + alphabet
        self.merges = {}
        
        # Initial splits: each word split into characters
        splits = {word: [c for c in word] for word in self.word_freqs.keys()}
        
        # Step 2: Perform BPE merges
        while len(self.vocab) < vocab_size:
            pair_freqs = self.get_stats(splits)
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs, key=pair_freqs.get)
            max_freq = pair_freqs[best_pair]
            
            if max_freq == 0:
                break
                
            # Merge the best pair
            splits = self.merge_pair(*best_pair, splits)
            merged_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_token
            
            if merged_token not in self.vocab:
                self.vocab.append(merged_token)
            
            print(f"Merged {best_pair} -> '{merged_token}', vocab size: {len(self.vocab)}")
            
            # Early stopping if we're not making progress
            if len(self.vocab) % 50 == 0 and len(self.merges) < 10:
                print("Stopping early - not enough merges happening")
                break
        
        self.vocab_size = len(self.vocab)
        return self.vocab
    
    def tokenize(self, text):
        """Tokenize text into subword tokens"""
        # Pre-tokenize into words with space markers
        text = re.sub(r'\s+', ' ', text.strip())
        words = text.split(' ')
        
        tokens = []
        for i, word in enumerate(words):
            if not word:
                continue
                
            # Add space marker for all words except the first one
            if i > 0:
                word_to_tokenize = 'Ġ' + word
            else:
                word_to_tokenize = word
            
            # Start with individual characters
            split = list(word_to_tokenize)
            
            # Apply all learned merges
            changed = True
            while changed and len(split) > 1:
                changed = False
                i = 0
                while i < len(split) - 1:
                    pair = (split[i], split[i + 1])
                    if pair in self.merges:
                        split = split[:i] + [self.merges[pair]] + split[i + 2:]
                        changed = True
                    else:
                        i += 1
            
            # Convert to token IDs
            for token in split:
                if token in self.vocab:
                    tokens.append(self.vocab.index(token))
                else:
                    tokens.append(self.vocab.index("<|endoftext|>"))
        
        return tokens
    
    def detokenize(self, token_ids):
        """Convert token IDs back to text"""
        tokens = [self.vocab[i] for i in token_ids]
        
        text = ""
        for token in tokens:
            if token.startswith('Ġ'):
                if text:  # Only add space if we already have text
                    text += ' ' + token[1:]
                else:
                    text += token[1:]  # First word, no space
            else:
                text += token
        
        return text

# Usage
tokenizer = SimpleBPETokenizer()

# Your training data
training_data = [
    'cats rule the world',
    'dogs are the best', 
    'elephants have long trunks',
    'monkeys like bananas',
    'pandas eat bamboo',
    'tigers are dangerous',
    'zebras have stripes',
    'lions are the kings of the savannah',
    'giraffes have long necks',
    'hippos are big and scary',
    'rhinos have horns',
    'penguins live in the arctic',
    'polar bears are white'
]

# Add your file data
files_to_load = [
    "data/short_snippets.txt",
    "data/sierra_data.txt", 
    "data/forsaken_data.txt",
    "data/dominion_rp_epd.txt",
    "data/dominion_rp_disestro.txt"
]

for file_path in files_to_load:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            training_data.append(content)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        continue

# Clean the data
training_data = [ftfy.fix_text(text) for text in training_data]

# Train the tokenizer
print("Training BPE tokenizer...")
vocab = tokenizer.train(training_data, vocab_size=1000)
print(f"Final vocabulary size: {len(vocab)}")

# Test it
sample_text = "The Third 'Herald' of the Order of Niven came from relatively simple beginnings"
print(f"\nOriginal: '{sample_text}'")

token_ids = tokenizer.tokenize(sample_text)
print(f"Token IDs: {token_ids}")
print(f"Tokens: {[tokenizer.vocab[i] for i in token_ids]}")

reconstructed = tokenizer.detokenize(token_ids)
print(f"Reconstructed: '{reconstructed}'")
print(f"Match: {reconstructed == sample_text}")

# Test with a simpler example first
print("\n=== Testing with simpler examples ===")
test_cases = [
    "hello world",
    "cats rule",
    "the quick brown fox"
]

for test_text in test_cases:
    token_ids = tokenizer.tokenize(test_text)
    reconstructed = tokenizer.detokenize(token_ids)
    print(f"'{test_text}' -> '{reconstructed}' (match: {test_text == reconstructed})")