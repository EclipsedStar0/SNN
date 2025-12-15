import re
from collections import Counter
from typing import List, Dict, Set

class VocabularyBuilder:
    def __init__(self, case_sensitive: bool = True):
        self.case_sensitive = case_sensitive
        self.vocab = {}
        self.reverse_vocab = {}
        self.next_id = 1
        
    def build_vocab_from_ngrams(self, text: str, min_ngram: int = 1, max_ngram: int = 4, 
                               min_freq: int = 2, top_k: int = 10000) -> Dict[int, str]:
        """
        Build vocabulary with case sensitivity option
        """
        text = self._preprocess_text(text)
        words = text.split()
        
        ngram_counter = Counter()
        
        for n in range(min_ngram, max_ngram + 1):
            for word in words:
                if len(word) >= n:
                    for i in range(len(word) - n + 1):
                        ngram = word[i:i + n]
                        ngram_counter[ngram] += 1
        
        common_ngrams = {ngram: count for ngram, count in ngram_counter.items() 
                        if count >= min_freq}
        
        sorted_ngrams = sorted(common_ngrams.items(), key=lambda x: x[1], reverse=True)
        top_ngrams = [ngram for ngram, count in sorted_ngrams[:top_k]]
        
        self._build_vocabulary(top_ngrams)
        return self.vocab
    
    def _preprocess_text(self, text: str) -> str:
        """Clean text while preserving case if needed"""
        if not self.case_sensitive:
            text = text.lower()
        
        # Remove non-alphanumeric but keep letters with diacritics
        #text = re.sub(r'[^\w\s]', '', text)
        #text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _build_vocabulary(self, tokens: List[str]):
        """Build vocabulary mapping"""
        self.vocab = {}
        self.reverse_vocab = {}
        self.next_id = 1
        
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        punctuation_tokens = [' ', ',', '.', '!', '?', '"', "'", '-', '+', '=', '*', '/', '%', '^', '\\', '\n', '\t', '`', '¬', '~', '`', '$', '£', '@', '#', '&', '(', ')', '[', ']', '{', '}', '<', '>', '|', ':', ';', '_']
        numeric_tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for entry in punctuation_tokens:
            special_tokens.append(entry)
        for entry in numeric_tokens:
            special_tokens.append(entry)
        
        
        for token in special_tokens:
            self.vocab[self.next_id] = token
            self.reverse_vocab[token] = self.next_id
            self.next_id += 1
        
        for token in tokens:
            if token not in self.reverse_vocab:
                self.vocab[self.next_id] = token
                self.reverse_vocab[token] = self.next_id
                self.next_id += 1
                VocabularyBuilder

if __name__ == "__main__":
    vocab_builder = VocabularyBuilder()
    with open("data/dominion_rp_disestro.txt", 'r') as file:
        content = file.read()
    
    vocabulary = vocab_builder.build_vocab_from_ngrams(content, min_ngram=1, max_ngram=4, top_k=1000)
    print(vocabulary)