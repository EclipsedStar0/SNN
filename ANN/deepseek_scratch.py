import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

@dataclass
class ModelConfig:
    """Configuration for the language model."""
    vocab_size: int
    max_seq_len: int = 128
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 2048  # 4 * d_model
    dropout: float = 0.1
    bias: bool = False,  # Modern practice: no bias in attention/ffn
    use_moe: bool = False,
    num_experts: int = 4,
    moe_top_k: int = 2
    
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

class SimpleBPETokenizer:
    """Byte-Pair Encoding tokenizer with improved efficiency."""
    
    SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    
    def __init__(self):
        self.vocab: List[str] = []
        self.merges: Dict[Tuple[str, str], str] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.word_freqs: Dict[str, int] = defaultdict(int)
        
    def train(self, texts: str, 
          target_compression: float = 0.5,
          min_frequency: int = 2,
          max_vocab_size: int = 50000,
          min_improvement: float = 0.001) -> List[str]:
        """Train BPE with accurate compression tracking."""
        
        # Build word frequencies and calculate true initial length
        initial_char_count = 0
        texts = re.sub(r'\s+', ' ', texts.strip())
        words = texts.split(' ')
        for i, word in enumerate(words):
            if not word:
                continue
            word_with_space = 'Ġ' + word if i > 0 else word
            self.word_freqs[word_with_space] += 1
        initial_char_count += len(texts)
        
        # Build alphabet
        alphabet = sorted(set(char for word in self.word_freqs for char in word))
        self.vocab = self.SPECIAL_TOKENS + alphabet
        splits = {word: list(word) for word in self.word_freqs}
        
        # Calculate initial token count (character-level tokens)
        current_token_count = sum(len(chars) * freq for word, (chars, freq) in 
                                 ((word, (splits[word], self.word_freqs[word])) 
                                  for word in self.word_freqs))
        print(current_token_count, initial_char_count)
        previous_compression = 1.0
        pbar = tqdm(desc="Training BPE")
        
        while len(self.vocab) < max_vocab_size:
            pair_freqs = self._get_pair_frequencies(splits)
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]
            
            # Calculate current compression ratio
            current_compression = current_token_count / initial_char_count
            #print(current_compression,' ')
            
            # Stopping criteria
            if best_freq < min_frequency:
                #print(best_freq, min_frequency)
                break
            if current_compression <= target_compression:
                #print(current_compression, target_compression)
                break
                
            # Check improvement
            improvement = previous_compression - current_compression
            #print(improvement)
            if improvement < min_improvement and len(self.vocab) > 1000:
                break
            
            # Perform merge and update token count accurately
            # Each merge reduces token count by the frequency (since we replace 2 tokens with 1)
            token_reduction = best_freq
            current_token_count -= token_reduction
            
            splits = self._merge_pair(*best_pair, splits)
            merged_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_token
            
            if merged_token not in self.vocab:
                self.vocab.append(merged_token)
            
            previous_compression = current_compression
            pbar.set_description(f"Vocab: {len(self.vocab)}, True Comp: {current_compression:.3f}")
        
        pbar.close()
        
        # Build dictionaries
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Verify actual compression
        actual_compression = self._measure_actual_compression(texts)
        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Training compression: {current_compression:.3f}")
        print(f"Actual compression: {actual_compression:.3f}")
        
        return self.vocab
    
    def _measure_actual_compression(self, texts: List[str]) -> float:
        """Measure the actual compression ratio on the training data."""
        total_chars = 0
        total_tokens = 0
        
        for text in texts:
            # Count original characters (without special space markers)
            clean_text = re.sub(r'\s+', ' ', text.strip())
            total_chars += len(clean_text)
            
            # Count tokens after tokenization
            tokens = self.encode(text)
            total_tokens += len(tokens)
        
        return total_tokens / total_chars if total_chars > 0 else 1.0
    
    def _get_pair_frequencies(self, splits: Dict[str, List[str]]) -> Dict[Tuple[str, str], int]:
        """Calculate frequency of adjacent token pairs."""
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) < 2:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def _merge_pair(self, a: str, b: str, splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge all occurrences of pair (a, b)."""
        for word in self.word_freqs:
            split = splits[word]
            if len(split) < 2:
                continue
            
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
        return splits
    
    def encode(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        text = re.sub(r'\s+', ' ', text.strip())
        words = text.split(' ')
        
        tokens = []
        for i, word in enumerate(words):
            if not word:
                continue
            
            word_to_tokenize = 'Ġ' + word if i > 0 else word
            split = list(word_to_tokenize)
            
            # Apply learned merges
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
            
            # Convert to IDs
            for token in split:
                tokens.append(self.token_to_id.get(token, self.token_to_id["<UNK>"]))
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.id_to_token[i] for i in token_ids if i in self.id_to_token]
        
        text = ""
        for token in tokens:
            if token in self.SPECIAL_TOKENS:
                continue
            if token.startswith('Ġ'):
                text += ' ' + token[1:] if text else token[1:]
            else:
                text += token
        
        return text.strip()
    
    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<PAD>"]
    
    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<BOS>"]
    
    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<EOS>"]
        
    def size(self):
        return len(self.dictionary)


class RoPEPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) - modern alternative to sinusoidal."""
    
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        
        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cached cos/sin values if needed."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin embeddings for the sequence."""
        seq_len = x.shape[1]
        self._update_cache(seq_len, x.device)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys."""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with modern optimizations."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        
        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_bias = nn.Parameter(torch.zeros(config.n_heads, config.max_seq_len, config.max_seq_len))
        
        # Flash attention flag (use when available)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Fused QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if provided
        if cos is not None and sin is not None:
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Attention with causal mask
        if self.flash:
            # Use Flash Attention if available (PyTorch 2.0+)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention implementation
            attn_weights = attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = attn_weights + self.attn_bias[:, :T, :T].unsqueeze(0)
            
            # Apply causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            
            # Apply padding mask if provided
            if mask is not None:
                mask = mask.view(B, 1, 1, T)
                attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = attn_weights @ v
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(attn_output))

class MoEFFN(nn.Module):
    """Mixture of Experts Feed-Forward Network"""
    
    def __init__(self, config: ModelConfig, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Experts
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(config.d_model, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Get gating scores
        gates = self.gate(x)  # [B, T, num_experts]
        
        # Top-k routing
        top_k_weights, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Expert computation
        for i, expert in enumerate(self.experts):
            # Create mask for this expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            
            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_output = expert(expert_input)
                
                # Get weights for this expert
                expert_weights = top_k_weights[expert_mask]
                expert_weights = expert_weights[top_k_indices[expert_mask] == i].unsqueeze(-1)
                
                output[expert_mask] += expert_output * expert_weights
        
        return output
    

class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation (modern alternative to ReLU)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        # SwiGLU needs 2/3 * 4 * d_model for same param count
        hidden_dim = int(8 * config.d_model / 3)
        hidden_dim = int(2 * hidden_dim / 3)
        
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.w3 = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(W1*x) * (W2*x)
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    """Transformer decoder block with pre-normalization."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        if config.use_moe:
            self.ffn = MoEFFN(config, num_experts=config.num_experts, top_k=config.moe_top_k)
        else:
            self.ffn = FeedForward(config)
        
        # Learnable residual scaling
        self.alpha_attn = nn.Parameter(torch.ones(1))
        self.alpha_ffn = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.alpha_attn * self.attn(self.ln1(x), mask, cos, sin)
        x = x + self.alpha_ffn * self.ffn(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    """Modern transformer language model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding (RoPE)
        self.rope = RoPEPositionalEncoding(config.d_model // config.n_heads, config.max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Output head (tied with input embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight  # Weight tying
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('w3.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/np.sqrt(2 * config.n_layers))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, idx: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = idx.shape
        
        # Token embeddings with scaling
        x = self.token_embedding(idx) * (self.config.d_model ** 0.5)
        x = self.dropout(x)
        
        # Get RoPE embeddings
        cos, sin = self.rope(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask, cos, sin)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'tokenizer_vocab': tokenizer.vocab,
            'tokenizer_merges': tokenizer.merges,
            'tokenizer_word_freqs': tokenizer.word_freqs
        }, path)
        print(f'Checkpoint saved to {path}')
    
    @staticmethod
    def load_checkpoint(path: str, device: str = 'cpu') -> 'LanguageModel':
        """Load model from checkpoint."""
        with torch.serialization.safe_globals([ModelConfig]):
            checkpoint = torch.load(path, map_location=device)
        model = LanguageModel(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        tokenizer = SimpleBPETokenizer()
        tokenizer.vocab = checkpoint['tokenizer_vocab']
        tokenizer.merges = checkpoint['tokenizer_merges']
        tokenizer.word_freqs = checkpoint['tokenizer_word_freqs']
        
        tokenizer.token_to_id = {token: idx for idx, token in enumerate(tokenizer.vocab)}
        tokenizer.id_to_token = {idx: token for token, idx in tokenizer.token_to_id.items()}
        
        return model.to(device), tokenizer.to(device)


class TextDataset(Dataset):
    """Dataset for language modeling."""
    
    def __init__(self, texts:str, seq_len: int, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.sequences = []
        
        print("Splitting into chunks")
        chunks = texts.split("<SEP>")
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 0]
        print(f"We have {len(chunks)} chunks perform sliding window on")
        
        self.sequences = []
        chunk_token_ids = []
        for chunk in tqdm(chunks, desc="Tokenizing chunks"):
            token_ids = self.tokenizer.encode(chunk)  # This encodes the entire chunk
            chunk_token_ids.append(token_ids)
        
        # Boundary-aware approach
        boundary_sequences = 0
        boundary_tokens = 0
        
        for token_ids in chunk_token_ids:
            if len(token_ids) >= seq_len:
                num_sequences_in_chunk = len(token_ids) - seq_len
                boundary_sequences += num_sequences_in_chunk
                
                for i in range(num_sequences_in_chunk):
                    sequence = token_ids[i:i + seq_len + 1]
                    self.sequences.append(sequence)
                    boundary_tokens += len(sequence)  # Each sequence has seq_len+1 tokens
        
        # Calculate contiguous approach for comparison
        all_tokens = self.tokenizer.encode(texts)
        contiguous_sequences = max(0, len(all_tokens) - seq_len)
        contiguous_tokens = contiguous_sequences * (seq_len + 1)  # Each sequence has seq_len+1 tokens
        
        print(f"Boundary-aware: {boundary_sequences} sequences, {boundary_tokens} tokens")
        print(f"Contiguous: {contiguous_sequences} sequences, {contiguous_tokens} tokens")
        print(f"Reduction: {((contiguous_sequences - boundary_sequences) / contiguous_sequences * 100):.1f}% fewer sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        chunk = self.sequences[idx]
        
        # Pad if needed (should rarely happen with proper chunk filtering)
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [self.tokenizer.pad_token_id] * (self.seq_len + 1 - len(chunk))
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        mask = (x != self.tokenizer.pad_token_id).long()
        
        return x, y, mask

class MemoryEfficientTrainer:
    """Training loop with memory optimizations and mixed precision."""
    
    def __init__(self, model: LanguageModel, tokenizer: SimpleBPETokenizer, config: ModelConfig,
                 learning_rate: float = 3e-4, weight_decay: float = 5e-5, device: str = 'cuda',
                 gradient_checkpointing: bool = True):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Enable memory efficient attention
        if hasattr(F, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Gradient checkpointing for larger models
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Optimizer with weight decay (AdamW)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for transformer blocks."""
        def make_checkpointed_forward(block):
            original_forward = block.forward
            
            def checkpointed_forward(x, mask=None, cos=None, sin=None):
                return torch.utils.checkpoint.checkpoint(
                    original_forward, x, mask, cos, sin, use_reentrant=False
                )
            
            return checkpointed_forward
        
        for block in self.model.blocks:
            block.forward = make_checkpointed_forward(block)
    
    def train(self, train_data: str, epochs: int, batch_size: int, 
              eval_interval: int = 600, save_interval: int = 2000):
        """Train the model with mixed precision."""
        # Tokenize all data
        # Create dataset and dataloader
        dataset = TextDataset(train_data, self.config.max_seq_len, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Learning rate scheduler with warmup
        warmup_steps = len(dataloader) * 2
        total_steps = len(dataloader) * epochs
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Training loop
        self.model.train()
        global_step = 0
        losses = []
        last_save_time = time.time()
        last_eval_time = last_save_time
        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            epoch_losses = []
            
            for x, y, mask in pbar:
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    logits = self.model(x, mask)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                
                # Mixed precision backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scheduler.step()
                
                # Logging
                epoch_losses.append(loss.item())
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}', 
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                global_step += 1
                
                # Generate sample
                if global_step % eval_interval == 0 or time.time() - last_eval_time >= 1800:
                    self._generate_sample()
                    last_eval_time = time.time()
                
                # Save checkpoint
                if global_step % save_interval == 0 or time.time() - last_save_time >= 1800:
                    self.model.save_checkpoint(f'models/text_completion_23_D_mini_checkpoint_step_{global_step}.pth')
                    last_save_time = time.time()
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}')
            
            # Generate samples at end of epoch
            self._generate_sample()
        
        return losses
    
    @torch.no_grad()
    def _generate_sample(self):
        """Generate and print a sample during training."""
        self.model.eval()
        
        prompts = [
            #"The cat",
            #"As the storm came in from",
            "Deep within the Delvos Mountains", 
            #"The third 'Herald' of the Order of Niven",
            "of the",
            "it has"
        ]
        
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            if not tokens:  # Handle empty tokenization
                continue
                
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            try:
                output = self.model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=50)
                text = self.tokenizer.decode(output[0].tolist())
                
                print(f'\nPrompt: "{prompt}"')
                print(f'Generated: {text}')
            except Exception as e:
                print(f'Error generating for prompt "{prompt}": {e}')
        
        self.model.train()



def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total:,}')
    return total


def load_txt_files_with_pathlib(directory_path):
    """
    Load all .txt files using pathlib (supports recursive search).
    """
    training_data = []
    directory = Path(directory_path)
    
    # For recursive search:
    for file_path in directory.rglob("*.txt"):
        try:
            content = file_path.read_text(encoding='utf-8')
            training_data.append(content)
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return training_data

# Example usage
if __name__ == "__main__":
    # Sample data
    small_training_data = [
        """<BOS>cats rule the world.,
        dogs are the best., 
        elephants have long trunks.,
        monkeys like bananas.
        pandas eat bamboo.
        tigers are dangerous.
        zebras have stripes.
        lions are the kings of the savannah.
        giraffes have long necks.
        hippos are big and scary.
        rhinos have horns.
        penguins live in the arctic.
        polar bears are white.<EOS>""",
        '<SEP>',
        """<BOS>Modern-day Terra can summarily be divided into 12 distinct bodies, of which, seven are continents, and 5 are oceans. East of the international dateline, going from the north hemisphere to the southern hemisphere before moving east, past the sprawling Pacific Ocean-- the first continent to be remarked upon is North America. Home to the United States of America, Canada, Mexico, and a sprinkling of smaller hispanic countries in Central America (the 'bridge' between North and South America), North America is considerably developed and has 618.8 million inhabitants as of November 2025. South America, the North's southern neighbour, lies to the south, beneath the equator. Home to nations such as Brazil, Columbia, and Argentine, over 439 million people live in the continent (as of 2025), of which, nearly half of the continent's population lives in Brazil. Directly across from South America, across the second largest of Terra's oceans-- the Atlantic Ocean, is Africa. A vast continent known both for its deserts and its savannahs, Africa has a long, tragic history. Exploited and ravaged by more developed locales, enslaved and pillaged by imperialists and colonial powers, despite hosting 1.53 Billion humans-- a full 18% of humanity as of 2025, Africa remains one of the most impoverished, underdeveloped continents in the world with hundreds of millions living in extreme poverty. 

Europe... Europe on the other hand, often collectively referred to as the 'West' (with the US and Canda included often times), is highly developed. The most developed continent in the world, 725.8 million people reside within Europe. For nearly half a millenia, Europe has been dominant, enslaving, pillaging, exterminating, and exploiting native populations in other continents, a trend that only began to decline in the last century, ending primarily from the World Wars and the wave of decolonization that spread across the globe. France, England, Spain, Germany, and Italy, are amongst the powers of Europe.

Across from Europe, demarked by the Baltic, Ukrainian, and Turkish border, is Asia. The most populous continent in the world with 4.97 billion people, making up nearly 60% of the global population, Asia is home to the Russian Federation, the People's Republic of China, the Republic of India, and numerous smaller nations. Interestingly enough, the Republic of India is situated on the Indian Subcontinent, a massive outcropping of land demarked by the Himalaya mountains to the north, with the nation of Nepal inhabitting the mountainous region, seperate from both China and India, amongst the two dominant great powers of Asia.

Further to the south of Asia, across the Indian Ocean, is Oceania, the sixth of the seven continents. A vast archipalego of thousands of islands big and small, Oceania only plays host to approximately 44 million inhabitants-- a paltry sum of the 8.259 billion people living on the Earth as of 2025... making up not even a single percentage point of the world's population.

And then, at the southern pole of the Earth, is Antarctica, the Earth's southernmost continent, it has the smallest population of all seven continents despite being 40% larger than Europe with a population numbering at several thousand during the summer wonths-- a number that plummets to a mere thousand come winter. On the other end of the globe, on the Earth's north pole, is the Arctic Circle a frozen collection of seas that upon which the northern tips of North America, Europe, and Asia converge upon.<EOS><SEP>""",
    ]
    
    # Add your file data
    #files_to_load = [
    #    "data/dominion_rp_epd.txt",
    #    "data/dominion_rp_disestro.txt",
    #    "data/dominion_rp_electua_solo_only.txt",
    #    "data/worm_mini_non_fanfic.txt",
    #    "data/sierra_data.txt", 
    #    "data/forsaken_data.txt",
    #    "data/short_snippets.txt",
    #    "data/additional_stories.txt",
    #    "data/merek_vr_ravenfield.txt",
    #    "data/mini_litrpg_abomination.txt"
    #]
    #
    #for file_path in files_to_load:
    #    try:
    #        with open(file_path, 'r', encoding='utf-8') as file:
    #            content = file.read()
    #            training_data.append('<BOS>\n'+content+'\n<EOS><SEP>')
    #    except FileNotFoundError:
    #        print(f"Warning: {file_path} not found")
    #        continue
    
    training_data = load_txt_files_with_pathlib("data")
    training_data.extend(small_training_data)
    training_data = [ftfy.fix_text(text) for text in training_data]
    training_data = "".join(training_data)
            
    # Train tokenizer
    print("Training tokenizer...")
    tokenizer = SimpleBPETokenizer()
    tokenizer.train(
        training_data,
        target_compression = 0.6, # was 0.35
        min_frequency = 2,
        max_vocab_size = 6000,
        min_improvement = 0.00005
    )
    print(tokenizer.vocab)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    tData = "".join(training_data)
    print(len(tData))
    print(len(tokenizer.encode(tData)))
    
    config = ModelConfig(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=128, # was 256
        d_model=512,
        n_layers=4, # was 6
        n_heads=2, # was 4
        dropout=0.1,
        d_ff = 2048,
        use_moe = True,  # Add this
        num_experts = 0,  # Was 4
        moe_top_k = 0    # Was 2
        # NOTE: increased dropout = increased loss; 0.3 has double loss of 0.2; 0.2 lags around 10/20 Epochs behing 0.1
        # 512 * 8 * 4 * 2048
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Create model
    
    model = LanguageModel(config).to(device)
    count_parameters(model)
    
    # Train
    trainer = MemoryEfficientTrainer(
        model, 
        tokenizer, 
        config, 
        device=device,
        learning_rate=3e-4,
        gradient_checkpointing=True  # Enable for larger models
    )
    
    losses = trainer.train(
        training_data, 
        epochs=12, 
        batch_size=512,
        eval_interval=600,
        save_interval=2000
    )
    
    # Save final model
    model.save_checkpoint('models/text_completion_23_D_mini_Final.pth')