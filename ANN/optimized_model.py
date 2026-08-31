"""
Optimized Language Model Training Script
Incorporating key optimizations from state-of-the-art implementations
"""
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

PAD_TOKEN = "<PAD>"
SEP_TOKEN = "<SEP>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(f"Using device: {device}")

MODEL_NAME = "unnamed_model"

@dataclass
class ModelConfig:
    """Configuration for the language model."""
    vocab_size: int
    max_seq_len: int = 256
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8  # Changed from 4 to 8 for better parallelism
    d_ff: int = 2048
    dropout: float = 0.1
    bias: bool = False
    use_moe: bool = False
    num_experts: int = 4
    moe_top_k: int = 2
    # New optimization flags
    use_flash_attn: bool = True
    use_rmsnorm: bool = True  # RMSNorm is faster than LayerNorm
    tie_embeddings: bool = True
    use_swiglu: bool = True  # SwiGLU activation
    additional_pref_suf: str = ''
    RoPE_freq:int = 10000
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - faster than LayerNorm."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x / rms(x) * weight
        # This is simpler and faster than LayerNorm
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class RoPEPositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos/sin values if needed."""
        if (seq_len != self._seq_len_cached) or self._seq_len_cached < seq_len or (self._cos_cached is None) or (self._cos_cached.device != device):
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin embeddings for the sequence."""
        seq_len = x.shape[1]
        self._update_cache(seq_len, x.device, x.dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


@torch.jit.script
def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, 
                     cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings - JIT compiled for speed."""
    # Split into pairs
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    
    # Apply rotation
    q_rot = torch.stack([
        q1 * cos[..., ::2] - q2 * sin[..., ::2],
        q1 * sin[..., 1::2] + q2 * cos[..., 1::2]
    ], dim=-1).flatten(-2)
    
    k_rot = torch.stack([
        k1 * cos[..., ::2] - k2 * sin[..., ::2],
        k1 * sin[..., 1::2] + k2 * cos[..., 1::2]
    ], dim=-1).flatten(-2)
    
    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        
        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        nn.init.zeros_(self.out_proj.weight)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Use Flash Attention if available
        self.flash = config.use_flash_attn and hasattr(F, 'scaled_dot_product_attention')
        
        # Attention scale
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Fused QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if provided
        if cos is not None and sin is not None:
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Apply QK normalization (from optimized code)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # Attention computation
        if self.flash:
            # Use Flash Attention (PyTorch 2.0+)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                scale=self.scale
            )
        else:
            # Manual attention
            attn_weights = (q @ k.transpose(-2, -1)) * self.scale
            
            # Causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
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


class SwiGLU(nn.Module):
    """SwiGLU activation - more powerful than standard FFN."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        # SwiGLU needs different hidden dimension
        hidden_dim = int(8 * config.d_model / 3)
        hidden_dim = int(2 * hidden_dim / 3)
        
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.w3 = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        
        # Zero init for output projection (from optimized code)
        nn.init.zeros_(self.w3.weight)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(W1*x) ⊗ (W2*x)
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class FeedForward(nn.Module):
    """Standard Feed-forward network (if not using SwiGLU)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        
        # Zero init for output projection
        nn.init.zeros_(self.w2.weight)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))

class MoEFFN(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        
        # Create experts
        if config.use_swiglu:
            self.experts = nn.ModuleList([SwiGLU(config) for _ in range(config.num_experts)])
        else:
            self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.num_experts)])
        
        # Gating network
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        
        self.load_balancing_loss = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        
        # Gating
        gates = self.gate(x_flat)
        top_k_weights, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Calculate load balancing loss
        if self.training:
            importance = F.softmax(gates, dim=-1).sum(0)
            load = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                load[i] = (top_k_indices == i).float().sum()
            
            importance = importance / importance.sum()
            load = load / (load.sum() + 1e-10)
            self.load_balancing_loss = (importance * load).sum() * self.num_experts
        
        # Route to experts
        output = torch.zeros_like(x_flat)
        
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            weights = top_k_weights[:, k:k+1]
            
            for expert_id in range(self.num_experts):
                mask = expert_indices == expert_id
                if mask.any():
                    expert_out = self.experts[expert_id](x_flat[mask])
                    output[mask] += expert_out * weights[mask]
        
        return output.view(B, T, C)


class TransformerBlock(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Use RMSNorm if enabled, otherwise LayerNorm
        norm_class = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln1 = norm_class(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = norm_class(config.d_model)
        
        # FFN or MoE
        if config.use_moe:
            self.ffn = MoEFFN(config)
        elif config.use_swiglu:
            self.ffn = SwiGLU(config)
        else:
            self.ffn = FeedForward(config)
        
        # Learnable residual scaling (from optimized code)
        self.alpha_attn = nn.Parameter(torch.ones(1))
        self.alpha_ffn = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (more stable)
        x = x + self.alpha_attn * self.attn(self.ln1(x), mask, cos, sin)
        x = x + self.alpha_ffn * self.ffn(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding (RoPE)
        self.rope = RoPEPositionalEncoding(config.d_model // config.n_heads, config.max_seq_len, config.RoPE_freq)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Final layer norm
        norm_class = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln_f = norm_class(config.d_model)
        
        # Output head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying (from optimized code)
        if config.tie_embeddings:
            self.token_embedding.weight = self.lm_head.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply scaled init to residual projections (from optimized code)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('w3.weight') or pn.endswith('w2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))
    
    def _init_weights(self, module):
        """Initialize weights - improved scheme from optimized code."""
        if isinstance(module, nn.Linear):
            # Use smaller std for initialization
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, idx: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = idx.shape
        
        # Token embeddings with scaling (from optimized code)
        x = self.token_embedding(idx)
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
        """Generate tokens autoregressively with optimizations."""
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
    
    def save_checkpoint(self, path: str, path_suffix: str, tokenizer: Tokenizer):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(f"models/{path}/"), exist_ok=True)
        torch.save({
            'config': self.config,
            'model_state_dict': self.state_dict()
        }, f"models/{path}/{path+path_suffix}"+'.pth')
        tokenizer.save(f"models/{path}/{path}_tokenizer{path_suffix}.json")
        print(f'Checkpoint saved to models/{path}')
    
    @staticmethod
    def load_checkpoint(path: str, path_suffix: str, device: str = 'cpu') -> Tuple['LanguageModel', Tokenizer]:
        """Load model from checkpoint."""
        with torch.serialization.safe_globals([ModelConfig]):
            checkpoint = torch.load(f"models/{path}/{path+path_suffix}"+'.pth', map_location=device, weights_only=False)
        model = LanguageModel(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        tokenizer = Tokenizer.from_file(f"models/{path}/{path}_tokenizer{path_suffix}.json")
        
        return model.to(device), tokenizer


class TextDataset(Dataset):
    
    def __init__(self, texts: str, seq_len: int, tokenizer, stride_len:int = 0, savethis:bool=False, loading_prev:bool=False, prior_name:str = "unnamed_model", moddable_suffix:str='training'):

        self.seq_len = seq_len
        self.stride_len = stride_len
        if self.stride_len == 0:
            self.stride_len = self.seq_len // 2
            
        self.tokenizer = tokenizer
        
        self.sequences = []
        self.loading_previous = loading_prev
        self.previous_model_name = prior_name
        self.save_this = savethis
        
        if texts == "Fake":
            return
            
        if not loading_prev:
            print("Splitting into chunks")
            chunks = [chunk.strip() for chunk in texts.split(SEP_TOKEN) if len(chunk.strip()) > 0]
            print(f"We have {len(chunks)} chunks to process")
            
            self.sequences = []
            chunk_token_ids = []
            size_of_base_text = 0
            
            # Tokenize all chunks at once (more efficient)
            for chunk in tqdm(chunks, desc="Tokenizing chunks"):
                encoded = self.tokenizer.encode(chunk)
                size_of_base_text += len(encoded.ids)
                chunk_token_ids.append(encoded.ids)
        
            print(f"Corpus is {size_of_base_text:,} tokens long")
            
            
            if self.save_this:
                print(f"Saving encoded corpus for later use...")
                with open(f'models/{MODEL_NAME}/{MODEL_NAME}_{moddable_suffix}_data.pkl', 'wb') as file:
                    pickle.dump(chunk_token_ids, file)
        
        if loading_prev:
            print("Loading an encoded corpus...")
            with open(f'models/{prior_name}/{prior_name}_{moddable_suffix}_data.pkl', 'rb') as file:
                chunk_token_ids = pickle.load(file)
        print(self.seq_len)
        print(self.stride_len)
        for token_ids in tqdm(chunk_token_ids, desc="Creating sequences"):
            if len(token_ids) >= self.seq_len:
                # Create overlapping sequences for better data efficiency
                for i in range(0, len(token_ids) - self.seq_len, self.stride_len):
                    sequence = token_ids[i:i + self.seq_len + 1]
                    if len(sequence) == self.seq_len + 1:
                        self.sequences.append(sequence)
            else:
                # Pad short sequences
                sequence = token_ids + [tokenizer.token_to_id(PAD_TOKEN)] * (self.seq_len + 1 - len(token_ids))
                self.sequences.append(sequence)
        
        print(f"Created {len(self.sequences):,} training sequences")
        print(f"We will be training on {len(self.sequences) * self.seq_len} tokens.")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        chunk = self.sequences[idx]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        mask = (x != self.tokenizer.token_to_id(PAD_TOKEN)).long()
        
        return x, y, mask
    
    def split_array_randomly(self, split_percentage, seed=None):
        """Split dataset into train/val."""
        if seed is not None:
            random.seed(seed)
        
        split_percentage = max(0, min(1, split_percentage))
        sequences_copy = self.sequences.copy()
        random.shuffle(sequences_copy)
        
        num_first_split = int(len(sequences_copy) * split_percentage)
        first_split = sequences_copy[:num_first_split]
        second_split = sequences_copy[num_first_split:]
        
        return first_split, second_split


class OptimizedTrainer:
    
    def __init__(self, model: LanguageModel, tokenizer: Tokenizer, config: ModelConfig,
                 learning_rate: float = 3e-4, cycle_length: int = 1000, weight_decay: float = 0.1, device: str = 'cuda',
                 gradient_checkpointing: bool = False, compile_model: bool = True):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.cycle_length = cycle_length
        self.max_lr = learning_rate
        self.weight_decay_var = weight_decay
        
        # Compile model for faster execution (PyTorch 2.0+)
        if device != 'cpu' and compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
        
        # Enable memory efficient attention
        if hasattr(F, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Gradient checkpointing (optional - trades compute for memory)
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Use AdamW with fused implementation (faster)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),  # From optimized code
            eps=1e-8,
            fused=True if device == 'cuda' else False
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id(PAD_TOKEN))
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda')
        self.use_amp = device == 'cuda'
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        def make_checkpointed_forward(block):
            original_forward = block.forward
            
            def checkpointed_forward(x, mask=None, cos=None, sin=None):
                return torch.utils.checkpoint.checkpoint(
                    original_forward, x, mask, cos, sin, use_reentrant=False
                )
            
            return checkpointed_forward
        
        for block in self.model.blocks:
            block.forward = make_checkpointed_forward(block)
    
    def train(self, training_start_time, train_data: str, test_data:str, epochs: int, batch_size: int,
              eval_interval: int = 600, save_interval: int = 2000, 
              gradient_accumulation_steps: int = 4, current_epoch: int = 0, stride_len:int = 1, clean_slate_save: bool = True, loading_prev:bool=False, prior_name:str = "unnamed_model"):
        """Train with all optimizations enabled."""
        
        config_info = f"{MODEL_NAME} Configuration = " + "{" +f"""
        \tSuffix = {self.config.additional_pref_suf}
\t"Layers" = {self.config.n_layers},
\t"Model Dim" = {self.config.d_model},
\t"FeedForward Dim" = {self.config.d_ff},
\t"Heads" = {self.config.n_heads},
\t"Weight Dropout" = {self.config.dropout},
\t"Vocab Size" = {self.config.vocab_size},
\t"Max Seq Len" = {self.config.max_seq_len},
\t"Stride Len" = {stride_len},
\t"RoPE_freq = {self.config.RoPE_freq},
\t"Weight Decay = {self.weight_decay_var},
\t"Max LR" = {self.max_lr},
\t"Batch Size" = {batch_size},
\t"Gradient Accumulation Steps" = {gradient_accumulation_steps},
\t"Eval Interval" = {eval_interval},
\t"Save Interval" = {save_interval},
\t"Max Epochs" = {epochs},
\t"Training Start Time" = {training_start_time}
""" + "}"
        
        
        os.makedirs(os.path.dirname(f"models/{MODEL_NAME}/"), exist_ok=True)
        with open(f'models/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_configuration.txt', 'w') as file:
            file.write(config_info)
        
        if clean_slate_save:    
            self.model.save_checkpoint(
                f'{MODEL_NAME}', 
                f'{self.config.additional_pref_suf}_CleanSlate', 
                self.tokenizer
            )
            
        # Create datasets
        dataset = TextDataset(train_data, self.config.max_seq_len, self.tokenizer, stride_len, clean_slate_save, loading_prev, prior_name, 'training')
        training_dataset = TextDataset("Fake", self.config.max_seq_len, self.tokenizer)
        validation_dataset = TextDataset("Fake", self.config.max_seq_len, self.tokenizer)
        testing_dataset = TextDataset(test_data, self.config.max_seq_len, self.tokenizer, stride_len, clean_slate_save, loading_prev, prior_name, 'testing')
        
        training_dataset.sequences, validation_dataset.sequences = dataset.split_array_randomly(0.95, 42)
        del dataset
        
        print(f"Training on: {len(training_dataset.sequences):,} sequences")
        print(f"Validating on: {len(validation_dataset.sequences):,} sequences")
        print(f"Testing on {len(testing_dataset.sequences):,} sequences")
        
        # Create dataloaders with optimized settings
        dataloader = DataLoader(
            training_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,  # Parallel data loading
            pin_memory=False,  # Faster data transfer to GPU
            persistent_workers=True,
            prefetch_factor=8
        )
        validation_dataloader = DataLoader(
            validation_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True,
            prefetch_factor=8
        )
        test_b_size = batch_size
        if len(testing_dataset.sequences) < batch_size:
            test_b_size /= 8
        test_b_size = int(test_b_size)
        testing_dataloader = DataLoader(
            testing_dataset, 
            batch_size=test_b_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True,
            prefetch_factor=8
        )
        
        # Cosine annealing with warmup (from optimized code)
        warmup_steps = (len(dataloader) / gradient_accumulation_steps) // 5
        total_steps = (len(dataloader) // gradient_accumulation_steps) * epochs
        
        # 0.5% (/100) is BAD -> Step 36000 - Val Loss: 4.9709
        # 5% (/10) performs better -> Step 24000 - Val Loss: 4.8579
        # 10% (/5) performs -?> 
        
        
        # def cycle_step(step):
        #     point_in_cycle = step % self.cycle_length
        #     percent_cycle = point_in_cycle / self.cycle_length
        #     
        #     return 0.5 + 0.5 * 0.5 * (1 + math.cos(math.pi * percent_cycle))
        #     
        #     
        # 
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        
        # scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, total_steps)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.7, patience=warmup_steps, cooldown=warmup_steps)
        
        
        # Training loop
        self.model.train()
        print(f"Initialization took {(time.time() - training_start_time):.2f}s")
        
        global_step = 0
        losses = []
        all_training_loss = []
        
        validation_losses = []
        testing_losses = []
        best_val_loss = float('inf')
        best_test_loss = float('inf')
        times_test_loss_has_worsened = 0
        total_times_test_loss_has_worsened = 0
        
        last_save_time = time.time()
        last_eval_time = last_save_time
        mandated_end_to_training = False
        
        for epoch in range(epochs-current_epoch):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1+current_epoch}/{epochs}")
            epoch_losses = []
            accumulated_loss = 0.0
            
            for batch_idx, (data_input, data_output, mask) in enumerate(pbar):
                data_input = data_input.to(self.device, non_blocking=True)
                data_output = data_output.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                
                
                loss = 0.0
                # Mixed precision forward pass
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(data_input, mask)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), data_output.view(-1))
                    
                    # Add MoE load balancing loss if applicable
                    if self.config.use_moe and self.config.num_experts > 0:
                        lb_loss = 0.0
                        for block in self.model.blocks:
                            if hasattr(block.ffn, 'load_balancing_loss') and block.ffn.load_balancing_loss is not None:
                                lb_loss += block.ffn.load_balancing_loss
                        loss = loss + 0.01 * lb_loss
                    
                    # Scale loss for gradient accumulation
                loss /= gradient_accumulation_steps
                accumulated_loss += loss
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer step with gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if self.use_amp:
                        # Unscale and clip gradients
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        # Optimizer step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    #scheduler.step(metrics=accumulated_loss)
                    
                    # Logging
                    epoch_losses.append(accumulated_loss.item())
                    all_training_loss.append(accumulated_loss.item())
                    
                    pbar.set_postfix({
                        'loss': f'{accumulated_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
                    accumulated_loss = 0.0
                    
                    global_step += 1
                    
                    # Evaluation
                    if global_step % eval_interval == 0 or time.time() - last_eval_time >= 1800:
                        val_loss = self._calculate_validation_loss(validation_dataloader)
                        test_loss = self._calculate_validation_loss(testing_dataloader)
                        validation_losses.append((global_step, val_loss))
                        testing_losses.append((global_step, test_loss))
                        os.makedirs(os.path.dirname(f'tracking/{MODEL_NAME}/'), exist_ok=True)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_step_{global_step}_avg_tr_loss.pkl', 'wb') as file:
                            pickle.dump(losses, file)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_step_{global_step}_tr_loss.pkl', 'wb') as file:
                            pickle.dump(all_training_loss, file)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_step_{global_step}_val_loss.pkl', 'wb') as file:
                            pickle.dump(validation_losses, file)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_step_{global_step}_test_loss.pkl', 'wb') as file:
                            pickle.dump(testing_losses, file)
                        print(f"\nStep {global_step} - Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
                        
                        # Generate samples
                        self._generate_sample()
                        
                        # Save best model
                        if (val_loss < best_val_loss):
                            best_val_loss = val_loss
                            self.model.save_checkpoint(f'{MODEL_NAME}', f'{self.config.additional_pref_suf}_best_val', self.tokenizer)
                        if (test_loss < best_test_loss):
                            best_test_loss = test_loss
                            times_test_loss_has_worsened = 0
                            self.model.save_checkpoint(f'{MODEL_NAME}', f'{self.config.additional_pref_suf}_best_test', self.tokenizer)
                        else:
                            times_test_loss_has_worsened += 1
                            total_times_test_loss_has_worsened += 1
                            if (test_loss > best_test_loss * 1.03):
                                print("\x1B[38;5;196m\t[WARNING]: TEST LOSS EXCEEDS BEST BY 3%!!!\x1B[38;5;252m")
                                mandated_end_to_training = True
                            elif times_test_loss_has_worsened >= 5:
                                print("\x1B[38;5;196m\t[WARNING]: TEST LOSS WORSENED 5x IN A ROW!!!\x1B[38;5;252m")
                                mandated_end_to_training = True
                            elif total_times_test_loss_has_worsened >= 8 and times_test_loss_has_worsened >= 2:
                                print("\x1B[38;5;196m\t[WARNING]: TEST LOSS HAS PLATEAUED!!!\x1B[38;5;252m")
                                mandated_end_to_training = True
                        
                        last_eval_time = time.time()
                    
                    # Periodic checkpoint
                    if global_step % save_interval == 0 or time.time() - last_save_time >= 1800 or mandated_end_to_training:
                        os.makedirs(os.path.dirname(f'tracking/{MODEL_NAME}/'), exist_ok=True)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_step_{global_step}_avg_tr_loss.pkl', 'wb') as file:
                            pickle.dump(losses, file)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_step_{global_step}_tr_loss.pkl', 'wb') as file:
                            pickle.dump(all_training_loss, file)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_step_{global_step}_val_loss.pkl', 'wb') as file:
                            pickle.dump(validation_losses, file)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_step_{global_step}_test_loss.pkl', 'wb') as file:
                            pickle.dump(testing_losses, file)
                        print(f"\nStep {global_step} - Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
                        self.model.save_checkpoint(
                            f'{MODEL_NAME}', 
                            f'{self.config.additional_pref_suf}_step_{global_step}', 
                            self.tokenizer
                        )
                        last_save_time = time.time()
                if mandated_end_to_training:
                    break
            if mandated_end_to_training:
                break
            
            # End of epoch
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            val_loss = self._calculate_validation_loss(validation_dataloader)
            test_loss = self._calculate_validation_loss(testing_dataloader)
            validation_losses.append((global_step, val_loss))
            testing_losses.append((global_step, test_loss))
            
            if (val_loss < best_val_loss):
                best_val_loss = val_loss
            if (test_loss < best_test_loss):
                best_test_loss = test_loss
                times_test_loss_has_worsened = 0
            else:
                times_test_loss_has_worsened += 1
                total_times_test_loss_has_worsened += 1
                if (test_loss > best_test_loss * 1.03):
                    print("\x1B[38;5;196m\t[WARNING]: TEST LOSS EXCEEDS BEST BY 3%!!!\x1B[38;5;252m")
                    mandated_end_to_training = True
                elif times_test_loss_has_worsened >= 5:
                    print("\x1B[38;5;196m\t[WARNING]: TEST LOSS WORSENED 5x IN A ROW!!!\x1B[38;5;252m")
                    mandated_end_to_training = True
                elif total_times_test_loss_has_worsened >= 8 and times_test_loss_has_worsened >= 2:
                    print("\x1B[38;5;196m\t[WARNING]: TEST LOSS HAS PLATEAUED!!!\x1B[38;5;252m")
                    mandated_end_to_training = True
            
            os.makedirs(os.path.dirname(f'tracking/{MODEL_NAME}/'), exist_ok=True)
            with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_step_{global_step}_avg_tr_loss.pkl', 'wb') as file:
                pickle.dump(losses, file)
            with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_step_{global_step}_tr_loss.pkl', 'wb') as file:
                pickle.dump(all_training_loss, file)
            with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_step_{global_step}_val_loss.pkl', 'wb') as file:
                pickle.dump(validation_losses, file)
            with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.config.additional_pref_suf}_step_{global_step}_test_loss.pkl', 'wb') as file:
                pickle.dump(testing_losses, file)
            
            print(f'\nEpoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')
            
            if not mandated_end_to_training:
                self._generate_sample()
        
            if mandated_end_to_training:
                break
        
        if mandated_end_to_training:
            print("\x1B[38;5;196m\t[WARNING]: ENDING TRAINING EARLY!!!\x1B[38;5;214m")
            print(f"\x1B[38;5;49m\tFinished with Best Test Loss of: \x1B[38;5;46m{best_test_loss}\x1B[38;5;214m")
        
        return losses, validation_losses, testing_losses
    
    def _calculate_validation_loss(self, validation_dataloader):
        """Calculate validation loss efficiently."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data_input, data_output, mask in validation_dataloader:
                data_input = data_input.to(self.device, non_blocking=True)
                data_output = data_output.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(data_input, mask)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), data_output.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 100:
                    break
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def _generate_sample(self):
        """Generate samples during training."""
        self.model.eval()
        
        prompts = [
            "On the dawn of the seventh day on the eleventh hour,",
            "There can only be one answer, and one answer only to the enemies of",
            '"Wait— wait we can talk about this!',
            "What is 13 plus 49?",
            "It can be thusly said, that the most despised of all",
            "On the topic of ",
            "We can thus conclude from these findings that",
            "Quickly, the architect scribbled in his notebook",
            "Carefully, the thief crept through the shadowed alleys of",
            "The cat",
            "As the storm came in from",
            "Deep within the mountains",
            "There was a",
            "In the making of the"
        ]
        
        prompts = [
            "Consider",
            "Lemme",
            "Narrate:",
            "Directive:",
            "Silpha",
            "Kaelen",
            "Eleris",
            "What",
            "How",
            "Now",
            "[",
            "<"
        ]
        
        
        for prompt in prompts:
            encoded = self.tokenizer.encode(prompt)
            if not encoded:
                continue
            
            input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(self.device)
            
            try:
                output = self.model.generate(
                    input_ids, 
                    max_new_tokens=100, 
                    temperature=0.8, 
                    top_k=50
                )
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




def grab_file_names_recursive(directory_path):
    file_name_arr_temp = get_file_names(directory_path)
    file_name_arr = []
    for file_name in file_name_arr_temp:
        file_name_arr.append(f"{directory_path}/{file_name}")
    
    top_folders = get_folder_names(directory_path)
    for folder_name in top_folders:
        result = grab_file_names_recursive(f"{directory_path}/{folder_name}")
        if len(file_name_arr) > 0:
            file_name_arr.extend(result)
        elif len(result) > 0:
            file_name_arr = result
    
    return file_name_arr
    
def load_files_random(directory_path, max_files=600, seed=52):
    file_name_arr = grab_file_names_recursive(directory_path)

    random.seed(seed)
    random.shuffle(file_name_arr)
    chosen_files = []
    data = []
    num_files_chosen = 0
    for file_name in file_name_arr:
        if num_files_chosen >= max_files:
            break
        if file_name.endswith('.txt'):    
            try:
                with open(file_name, 'r', encoding='utf-8') as file:
                    content = file.read()
                    data.append(f'{BOS_TOKEN}\n{content}\n{EOS_TOKEN}{SEP_TOKEN}')            
                    chosen_files.append(file_name)
                    num_files_chosen += 1
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue    
    print(chosen_files)
    return data

def load_txt_files_with_pathlib(directory_path, max_files=600, seed=52):
    """Load text files from directory."""
    function_training_data = []
    directory = Path(directory_path)
    
    number_of_files_searched = 0
    for file_path in directory.rglob("*.txt"):
        if number_of_files_searched >= max_files:
            break
        try:
            content = file_path.read_text(encoding='utf-8')
            function_training_data.append(f'{BOS_TOKEN}\n{content}\n{EOS_TOKEN}{SEP_TOKEN}')
            print(f"Loaded: {file_path}")
            number_of_files_searched += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return function_training_data


def train_tokenizer(vocab_size, special_tokens, training_data, train=True):
    """Train BPE tokenizer."""
    bpe_trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
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



lorem_ipsum_placeholder = """

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis felis ex, sollicitudin id lacinia id, volutpat suscipit magna. Ut sit amet nibh diam. Morbi hendrerit mi vel erat gravida convallis. Nullam accumsan lacinia placerat. Sed ut turpis felis. Nulla non accumsan lacus. Proin elit mi, hendrerit nec molestie nec, bibendum non ipsum. Etiam ullamcorper mollis ornare. Fusce eleifend dictum dictum. Nulla imperdiet massa leo, at accumsan mi cursus vitae. Nulla laoreet erat at lectus tristique, in blandit nunc semper. Duis libero est, vulputate vel eleifend at, pellentesque blandit lectus.

Proin at sollicitudin justo. Cras varius venenatis dolor sed imperdiet. Nulla ornare commodo posuere. Morbi finibus orci dolor, quis varius massa feugiat et. Nullam iaculis posuere ornare. Sed ultricies, sem sit amet semper placerat, arcu justo rutrum lorem, in vulputate orci quam mollis quam. Sed ornare nisi lobortis purus suscipit, non interdum justo imperdiet. Morbi vitae semper urna. Suspendisse eu viverra risus.

Nulla ut urna facilisis justo maximus finibus in eu libero. Integer ut pharetra odio. In pulvinar mi at dictum luctus. Sed porttitor, mauris ac dignissim iaculis, arcu risus luctus enim, vel fermentum justo ligula at neque. Cras quis ornare orci. Sed varius risus lacus, accumsan commodo turpis lobortis id. Integer quis nibh non urna consequat egestas. Vivamus ut lobortis nisl. Nullam ut dui arcu. Vestibulum a faucibus leo, ac porta odio. Etiam ac metus eu lectus lobortis posuere. Aliquam pellentesque sem a augue malesuada imperdiet. Pellentesque convallis vulputate metus, at convallis ante ultricies nec. Fusce sit amet tellus dolor. Fusce vel augue sit amet dolor malesuada pretium.

Phasellus lacinia mauris massa, vitae gravida mi malesuada id. Donec interdum risus arcu, et tincidunt elit bibendum at. Praesent lorem turpis, aliquet eu lorem id, vestibulum congue nibh. Quisque tincidunt diam dignissim velit malesuada, vitae ultricies justo blandit. Pellentesque pulvinar, enim lobortis semper lobortis, magna risus posuere sem, eu auctor metus nisl et augue. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Quisque mi purus, vulputate id sollicitudin ac, fringilla non purus. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nulla venenatis tincidunt diam, eu pharetra eros pretium sit amet.

Suspendisse ac commodo nunc. Quisque ac ligula luctus, dignissim urna vitae, varius leo. Donec tristique semper lectus eu tincidunt. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ornare pulvinar dignissim. Praesent dignissim eu orci ultrices bibendum. Mauris metus est, sodales et posuere ut, dictum quis felis. Mauris consectetur quis risus ut pellentesque. Vivamus ac auctor orci. Vivamus pellentesque libero ut libero consectetur, sit amet rhoncus tellus lobortis. Aenean luctus libero sed sem gravida porta. Integer rhoncus purus risus, non mollis sapien molestie quis. Proin hendrerit ex ac tellus scelerisque, sed ornare tellus cursus. """





if __name__ == "__main__":
    
    # Defaults
    num_layers = 2
    embed_dim = 16
    ffn_dim = embed_dim * 4
    num_heads = 1
    weight_dropout = 0.1
    training_rope_freq = 10000
    
    max_vocab_len = 1000
    sequence_len = 16
    stride_length = sequence_len // 2
    personal_files = 80
    data_files = 0
    sm_data_include = True
    LoC_data_include = False
    
    wei_decay = 0.1
    num_batches = 32
    num_grad_step = 32
    maximum_lr = 3e-4
    num_epochs = 30
    eval_iter = 2000
    save_iter = 2000
    
    operation_selection = prompt_select(
        "Where do you wish to begin?",
        choices=[
            Choice(
                title="From a 'CleanSlate'",
                value="clean",
            ),
            Choice(
                title="Continue In-Progress Training",
                value="continue",
            ),
            Choice(
                title="Begin New Training Run",
                value="new_model",            
            ),
        ],
    )
    
    if operation_selection == "clean":
        PRIOR_MODEL_NAME = "unnamed_model"
        
        choices = []
        folder_names = get_folder_names("models")
        
        if len(folder_names) < 1:
            raise Exception("No prior models. Closing")
        for index in range(len(folder_names)):
            choices.append(Choice(title=folder_names[index], value=folder_names[index]))
        PRIOR_MODEL_NAME = prompt_select("Which model-structure do you want to use?", choices)

        
        print("\x1B[38;5;123mTRAINING:\x1B[38;5;252m")
        sequence_len = int(input("\x1B[38;5;252m\tSequence Length: \x1B[38;5;214m"))
        stride_length = int(input("\x1B[38;5;252m\tStride Length: \x1B[38;5;214m"))
        training_rope_freq = int(input("\x1B[38;5;252m\tRoPE Freq: \x1B[38;5;214m"))
        wei_decay = float(input("\x1B[38;5;252mWeight Decay: \x1B[38;5;214m"))
        num_batches = int(input("\x1B[38;5;252mHow many sequences should be in each batch?: \x1B[38;5;214m"))
        num_grad_step = int(input("\x1B[38;5;252mHow many gradient accumulation steps?: \x1B[38;5;214m"))
        maximum_lr = float(input("\x1B[38;5;252mMaximum LR: \x1B[38;5;214m"))
        num_epochs = int(input("\x1B[38;5;252mNum Epochs: \x1B[38;5;214m"))
        eval_iter = int(input("\x1B[38;5;252mEvaluation Interval (Steps): \x1B[38;5;214m"))
        save_iter = int(input("\x1B[38;5;252mSave Interval (Steps): \x1B[38;5;214m"))
        MODEL_NAME = input("\x1B[38;5;252mModel Name: \x1B[38;5;214m")
        print("\x1B[38;5;252m")

        init_start_time = time.time()
        config = ModelConfig(
            vocab_size=max_vocab_len,
            max_seq_len=sequence_len,
            d_model=embed_dim,
            n_layers=num_layers,
            n_heads=num_heads, 
            d_ff=ffn_dim,
            dropout=weight_dropout,
            use_moe=False,
            num_experts=4,
            moe_top_k=2,
            use_flash_attn=True,
            use_rmsnorm=True,
            tie_embeddings=True,
            use_swiglu=True,
            RoPE_freq=training_rope_freq
        )
        
        model = LanguageModel(config).to(device)
        model, tokenizer = model.load_checkpoint(PRIOR_MODEL_NAME, '_CleanSlate')
        
        trainer = OptimizedTrainer(
            model,
            tokenizer,
            config,
            device=device,
            learning_rate=maximum_lr,
            cycle_length=1000,
            weight_decay=wei_decay,
            gradient_checkpointing=False,  # Set to True if running out of memory
            compile_model=True  # Enable torch.compile for speed
        )  
        
        losses, validation_losses, testing_data, = trainer.train(
            init_start_time,
            lorem_ipsum_placeholder,
            lorem_ipsum_placeholder,
            epochs=num_epochs,
            batch_size=num_batches,
            eval_interval=eval_iter,
            save_interval=save_iter,
            gradient_accumulation_steps=num_grad_step, # MAKE SURE BATCH_SIZE * GRADIENT STEPS IS 256 OR SMTH REASONABLE
            stride_len=stride_length,
            clean_slate_save=False,
            loading_prev=True,
            prior_name=PRIOR_MODEL_NAME,
        )
        print("Saving final model...")
        model.save_checkpoint(f'{MODEL_NAME}', '_final', tokenizer)
        
        print(f"\nTraining complete! Total time: {(time.time() - init_start_time)/60:.2f} minutes")
            
            
    elif operation_selection == "continue":
        choices = []
        folder_names = get_folder_names("models")
        
        if len(folder_names) < 1:
            raise Exception("No prior models. Closing")
        for index in range(len(folder_names)):
            choices.append(Choice(title=folder_names[index], value=folder_names[index]))
        folder_name = prompt_select("Choose model folder", choices)
        
        file_names = get_file_names(f"models/{folder_name}")
        
        choices = []
        for name in file_names:
            if name.endswith('.pth'):
                choices.append(Choice(title=name, value=name))
        chosen_model = prompt_select("Choose model to continue from", choices)
        chosen_model = chosen_model.split(".pth")[0]
        
        print("\x1B[38;5;123mTRAINING:\x1B[38;5;252m")
        sequence_len = int(input("\x1B[38;5;252m\tSequence Length: \x1B[38;5;214m"))
        stride_length = int(input("\x1B[38;5;252m\tStride Length: \x1B[38;5;214m"))
        training_rope_freq = int(input("\x1B[38;5;252m\tRoPE Frequency: \x1B[38;5;214m"))
        wei_decay = float(input("\x1B[38;5;252mWeight Decay: \x1B[38;5;214m"))
        num_batches = int(input("\x1B[38;5;252mHow many sequences should be in each batch?: \x1B[38;5;214m"))
        num_grad_step = int(input("\x1B[38;5;252mHow many gradient accumulation steps?: \x1B[38;5;214m"))
        maximum_lr = float(input("\x1B[38;5;252mMaximum LR: \x1B[38;5;214m"))
        num_epochs = int(input("\x1B[38;5;252mNum Epochs: \x1B[38;5;214m"))
        eval_iter = int(input("\x1B[38;5;252mEvaluation Interval (Steps): \x1B[38;5;214m"))
        save_iter = int(input("\x1B[38;5;252mSave Interval (Steps): \x1B[38;5;214m"))
        print("\x1B[38;5;252m")
        
        additional_pref_suffix = ''
        suffix = chosen_model.split(folder_name)[1]
        change_model_name = prompt_select(
            "Change Model Name?",
            choices=[
                Choice(
                    title='No',
                    value=False,
                ),
                Choice(
                    title='Yes',
                    value=True,
                ),            
            ]
        )
        
        if change_model_name:
            MODEL_NAME = input("\x1B[38;5;252mModel Name: \x1B[38;5;214m")
            print("\x1B[38;5;252m")
        else:
            MODEL_NAME = folder_name   
            
            step_cnt = 0
            if suffix.endswith('_final.pth'):
                additional_pref_suffix = '_finalC'
            elif suffix.endswith('_best.pth'):
                additional_pref_suffix = '_bestC'
            else:
                temp_s = suffix.split('_step_')[1]
                step_cnt = temp_s.split('.pth')[0]
                additional_pref_suffix = f'_S{step_cnt}'
                
            suffix = suffix.split('.pth')[0]
    
        print(f"NOTE: MAKE SURE THAT THERE IS A 'FOLDER-NAME_training_data.pkl' FILE IN models/{folder_name}/")
        tr_present = prompt_select(
            f"Is {folder_name}_training_data.pkl present??",
            choices=[
                Choice(
                    title='Yes',
                    value=True,
                ),  
                Choice(
                    title='No',
                    value=False,
                ),          
            ]
        )
        if not tr_present:
            raise Exception("Can't continue training without training data")


        init_start_time = time.time()
        config = ModelConfig(
            vocab_size=max_vocab_len,
            max_seq_len=sequence_len,
            d_model=embed_dim,
            n_layers=num_layers,
            n_heads=num_heads, 
            d_ff=ffn_dim,
            dropout=weight_dropout,
            use_moe=False,
            num_experts=4,
            moe_top_k=2,
            use_flash_attn=True,
            use_rmsnorm=True,
            tie_embeddings=True,
            use_swiglu=True,
            additional_pref_suf=additional_pref_suffix,
            RoPE_freq=training_rope_freq
        )
        
        model = LanguageModel(config).to(device)
        
        
        model, tokenizer = model.load_checkpoint(folder_name, suffix)
        
        trainer = OptimizedTrainer(
            model,
            tokenizer,
            config,
            device=device,
            learning_rate=maximum_lr,
            cycle_length=1000,
            weight_decay=wei_decay,
            gradient_checkpointing=False,  # Set to True if running out of memory
            compile_model=True  # Enable torch.compile for speed
        )  
        
        losses, validation_losses, testing_losses = trainer.train(
            init_start_time,
            lorem_ipsum_placeholder,
            lorem_ipsum_placeholder,
            epochs=num_epochs,
            batch_size=num_batches,
            eval_interval=eval_iter,
            save_interval=save_iter,
            gradient_accumulation_steps=num_grad_step, # MAKE SURE BATCH_SIZE * GRADIENT STEPS IS 256 OR SMTH REASONABLE
            stride_len=stride_length,
            clean_slate_save=False,
            loading_prev=True,
            prior_name=folder_name,
        )
        print("Saving final model...")
        model.save_checkpoint(f'{MODEL_NAME}', f'{additional_pref_suffix}_final', tokenizer)
        
        print(f"\nTraining complete! Total time: {(time.time() - init_start_time)/60:.2f} minutes")
        
        
    elif operation_selection == "new_model":
        no_error = False
        
        
        
        no_error = prompt_select(
            "Use defaults, or select hyperparameters?",
            choices=[
                Choice(
                    title="Use Defaults",
                    value=True,
                ),
                Choice(
                    title="Pick-and-Choose Hyperparameters",
                    value=False,
                ),
            ],
        )
        
        while not no_error:
            try:
                print("\x1B[38;5;123mMODEL INITIALIZATION:\x1B[38;5;252m")
                num_layers = int(input("\x1B[38;5;252m\tLayers: \x1B[38;5;214m"))
                embed_dim = int(input("\x1B[38;5;252m\tEmbed Dim: \x1B[38;5;214m"))
                ffn_dim = embed_dim * 4
                num_heads = int(input("\x1B[38;5;252m\tHeads: \x1B[38;5;214m"))
                weight_dropout = float(input("\x1B[38;5;252m\tDropout: \x1B[38;5;214m"))
                
                print("\x1B[38;5;123mDATA INITIALIZATION:\x1B[38;5;252m")
                max_vocab_len = int(input("\x1B[38;5;252m\tVocab Size: \x1B[38;5;214m"))
                sequence_len = int(input("\x1B[38;5;252m\tSequence Length: \x1B[38;5;214m"))
                stride_length = int(input("\x1B[38;5;252m\tStride Length: \x1B[38;5;214m"))
                training_rope_freq = int(input("\x1B[38;5;252m\tRoPE Freq: \x1B[38;5;214m"))
                personal_files = int(input("\x1B[38;5;252m\tHow many personal /data files should we import for training?: \x1B[38;5;214m"))
                data_files = int(input("\x1B[38;5;252m\tHow many /data gutenburg files should we import for training?: \x1B[38;5;214m"))
                sm_data_include = prompt_select(
                    "Should we include small_training_data?: ",
                    choices=[
                        Choice(
                            title="Yes",
                            value=True,
                        ),
                        Choice(
                            title="No",
                            value=False,
                        ),
                    ],
                )
                
                LoC_data_include = prompt_select(
                    "Should we include digitised data from the Library of Congress?",
                    choices=[
                        Choice(
                            title="Yes",
                            value=True,
                        ),
                        Choice(
                            title="No",
                            value=False,
                        ),
                    ],
                )
                fetch_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                if LoC_data_include:
                    
                    use_def_loc = prompt_select(
                        "Use Default LoC Corpus",
                        choices=[
                            Choice(
                                title="Yes",
                                value=True,
                            ),
                            Choice(
                                title="No",
                                value=False,
                            ),
                        ],
                    )
                    
                    if use_def_loc:
                        fetch_count = [0, 0, 10000, 5000, 2500, 1000, 100, 50, 40, 30, 20, 10, 8, 4]
                    else:
                        
                        LoC_data_dis = f"""\tBooks between 0-200 words: 153
            Books between 200-1000 words: 1249
            Books between 1000-5000 words: 11383
            Books between 5000-10000 words: 16088
            Books between 10000-20000 words: 20038
            Books between 20000-40000 words: 23274
            Books between 40000-60000 words: 15280
            Books between 60000-80000 words: 11601
            Books between 80000-100000 words: 8294
            Books between 100000-200000 words: 11531
            Books between 200000-300000 words: 4956
            Books between 300000-500000 words: 3475
            Books between 500000-1000000 words: 1807
            Books between 1000000+ words: 712"""
                        
                        print("\x1B[38;5;252mThe Digitised Library of Congress has the following book-word distribution:\n"+LoC_data_dis)
                        
                        
                        fetch_count[0] = int(input("\x1B[38;5;252m\tX books from the 0-200 words section: \x1B[38;5;214m"))
                        fetch_count[1] = int(input("\x1B[38;5;252m\tX books from the 200-1000 words section: \x1B[38;5;214m"))
                        fetch_count[2] = int(input("\x1B[38;5;252m\tX books from the 1000-5000 words section: \x1B[38;5;214m"))
                        fetch_count[3] = int(input("\x1B[38;5;252m\tX books from the 5000-10000 words section: \x1B[38;5;214m"))
                        fetch_count[4] = int(input("\x1B[38;5;252m\tX books from the 10000-20000 words section: \x1B[38;5;214m"))
                        fetch_count[5] = int(input("\x1B[38;5;252m\tX books from the 20000-40000 words section: \x1B[38;5;214m"))
                        fetch_count[6] = int(input("\x1B[38;5;252m\tX books from the 40000-60000 words section: \x1B[38;5;214m"))
                        fetch_count[7] = int(input("\x1B[38;5;252m\tX books from the 60000-80000 words section: \x1B[38;5;214m"))
                        fetch_count[8] = int(input("\x1B[38;5;252m\tX books from the 80000-100000 words section: \x1B[38;5;214m"))
                        fetch_count[9] = int(input("\x1B[38;5;252m\tX books from the 100000-150000 words section: \x1B[38;5;214m"))
                        fetch_count[10] = int(input("\x1B[38;5;252m\tX books from the 150000-200000 words section: \x1B[38;5;214m"))
                        fetch_count[11] = int(input("\x1B[38;5;252m\tX books from the 200000-300000 words section: \x1B[38;5;214m"))
                        fetch_count[12] = int(input("\x1B[38;5;252m\tX books from the 500000-1000000 words section: \x1B[38;5;214m"))
                        fetch_count[13] = int(input("\x1B[38;5;252m\tX books from the 1000000+ words section: \x1B[38;5;214m"))
                    
                print("\x1B[38;5;123mTRAINING:\x1B[38;5;252m")
                wei_decay = float(input("\x1B[38;5;252mWeight Decay: \x1B[38;5;214m"))
                num_batches = int(input("\x1B[38;5;252mHow many sequences should be in each batch?: \x1B[38;5;214m"))
                num_grad_step = int(input("\x1B[38;5;252mHow many gradient accumulation steps?: \x1B[38;5;214m"))
                maximum_lr = float(input("\x1B[38;5;252mMaximum LR: \x1B[38;5;214m"))
                num_epochs = int(input("\x1B[38;5;252mNum Epochs: \x1B[38;5;214m"))
                eval_iter = int(input("\x1B[38;5;252mEvaluation Interval (Steps): \x1B[38;5;214m"))
                save_iter = int(input("\x1B[38;5;252mSave Interval (Steps): \x1B[38;5;214m"))
                MODEL_NAME = input("\x1B[38;5;252mModel Name: \x1B[38;5;214m")
                
                no_error = True
            except KeyboardInterrupt:
                print("\x1B[38;5;252mClosing...")
                raise Exception('Closing')
            except:
                print("\x1B[38;5;252mERROR. You entered something incorrectly. Restarting.")
    
    
        init_start_time = time.time()
        print(f"\x1B[38;5;123mBEGINING TRAINING FOR: {MODEL_NAME}\x1B[38;5;252m")
        
        
        # Load and prepare data
        print("Loading training data...")
        training_data = []
        training_data = load_files_random("data/Gutenburg", max_files=data_files)
        testing_data = []
        
        
        num_personal_files_read = 0
        file_names = get_file_names("data")
        for file_name in file_names:
            if num_personal_files_read >= personal_files:
                break
            
            if file_name.endswith('.txt'):
                with open(f"data/{file_name}", 'r', encoding='utf-8') as file:
                    content = file.read()
                    training_data.append(f'{BOS_TOKEN}\n{content}\n{EOS_TOKEN}{SEP_TOKEN}')
                    num_personal_files_read += 1
            elif file_name.endswith('.TEST'):
                with open(f"data/{file_name}", 'r', encoding='utf-8') as file:
                    content = file.read()
                    testing_data.append(f'{BOS_TOKEN}\n{content}\n{EOS_TOKEN}{SEP_TOKEN}')
                    num_personal_files_read += 1
        
        
        
        
        small_training_data = [
            f"""{BOS_TOKEN}cats rule the world.,
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
            polar bears are white.{EOS_TOKEN}""",
            f"{SEP_TOKEN}",
            f"""{BOS_TOKEN}Modern-day Terra can summarily be divided into 12 distinct bodies, of which, seven are continents, and 5 are oceans. East of the international dateline, going from the north hemisphere to the southern hemisphere before moving east, past the sprawling Pacific Ocean-- the first continent to be remarked upon is North America. Home to the United States of America, Canada, Mexico, and a sprinkling of smaller hispanic countries in Central America (the 'bridge' between North and South America), North America is considerably developed and has 618.8 million inhabitants as of November 2025. South America, the North's southern neighbour, lies to the south, beneath the equator. Home to nations such as Brazil, Columbia, and Argentine, over 439 million people live in the continent (as of 2025), of which, nearly half of the continent's population lives in Brazil. Directly across from South America, across the second largest of Terra's oceans-- the Atlantic Ocean, is Africa. A vast continent known both for its deserts and its savannahs, Africa has a long, tragic history. Exploited and ravaged by more developed locales, enslaved and pillaged by imperialists and colonial powers, despite hosting 1.53 Billion humans-- a full 18% of humanity as of 2025, Africa remains one of the most impoverished, underdeveloped continents in the world with hundreds of millions living in extreme poverty. 

    Europe... Europe on the other hand, often collectively referred to as the 'West' (with the US and Canda included often times), is highly developed. The most developed continent in the world, 725.8 million people reside within Europe. For nearly half a millenia, Europe has been dominant, enslaving, pillaging, exterminating, and exploiting native populations in other continents, a trend that only began to decline in the last century, ending primarily from the World Wars and the wave of decolonization that spread across the globe. France, England, Spain, Germany, and Italy, are amongst the powers of Europe.

    Across from Europe, demarked by the Baltic, Ukrainian, and Turkish border, is Asia. The most populous continent in the world with 4.97 billion people, making up nearly 60% of the global population, Asia is home to the Russian Federation, the People's Republic of China, the Republic of India, and numerous smaller nations. Interestingly enough, the Republic of India is situated on the Indian Subcontinent, a massive outcropping of land demarked by the Himalaya mountains to the north, with the nation of Nepal inhabitting the mountainous region, seperate from both China and India, amongst the two dominant great powers of Asia.

    Further to the south of Asia, across the Indian Ocean, is Oceania, the sixth of the seven continents. A vast archipalego of thousands of islands big and small, Oceania only plays host to approximately 44 million inhabitants-- a paltry sum of the 8.259 billion people living on the Earth as of 2025... making up not even a single percentage point of the world's population.

    And then, at the southern pole of the Earth, is Antarctica, the Earth's southernmost continent, it has the smallest population of all seven continents despite being 40% larger than Europe with a population numbering at several thousand during the summer wonths-- a number that plummets to a mere thousand come winter. On the other end of the globe, on the Earth's north pole, is the Arctic Circle a frozen collection of seas that upon which the northern tips of North America, Europe, and Asia converge upon.{EOS_TOKEN}""",
        ]
        
        if sm_data_include:
            print("Extending Non-File data...")
            training_data.extend(small_training_data)
            
        
        if LoC_data_include:
            #ds = datasets.load_from_disk("hf_dataset\LoC-PD-Books")
            #for index in range(600):
            #    partial_sample = ds['train'][index]['text']# Optional: Add dataset
            #    training_data.append(f"{BOS_TOKEN}{partial_sample}{EOS_TOKEN}{SEP_TOKEN}")
            
            # ds = datasets.load_from_disk("hf_dataset/LoC-PD-Books")
            # low_tol = 1000
            # high_tol = 40000
            # 
            # 
            # total_entries = len(ds['train'])
            # print(f'High Tolerance: {high_tol}')
            # print(f'Low Tolerance: {low_tol}')
            # print(f'Total Number of Entries: {total_entries}')
            # 
            # progress_bar = tqdm(total=total_entries, desc="Scanning books...")
            # for index in range(total_entries):
            #     lenI = len(ds['train'][index]['text'])
            #     if (lenI > low_tol and lenI < high_tol):
            #         lengths.append(lenI)
            #     progress_bar.update(1)  # Update the progress bar
            # 
            # # Update the print statement to reflect the count of lengths, not lenI
            # print(f'Number within tolerance: {len(lengths)}')
            # print(f'Mean: {numpy.mean(lengths):0.0f}')
            # print(f'Stdv: {numpy.std(lengths):0.0f}')
            
            ds = datasets.load_from_disk("hf_dataset/LoC-PD-Books")
            indexes = {}
            
            # NOTE:
            # LoC Books between 0-1000 characters: 85
            # LoC Books between 1000-5000 characters: 1033
            # LoC Books between 5000-10000 characters: 1692
            # LoC Books between 10000-20000 characters: 4754
            # LoC Books between 20000-40000 characters: 11815
            # LoC Books between 40000-60000 characters: 10519
            # LoC Books between 60000-80000 characters: 8165
            # LoC Books between 80000-100000 characters: 6551
            # LoC Books between 100000-200000 characters: 12964
            # LoC Books between 200000-300000 characters: 9853
            # LoC Books between 300000-500000 characters: 14664
            # LoC Books between 500000-1000000 characters: 20272
            # LoC Books with 1000000+ characters: 19212
            
            # NOTE:
            # (C0 ) Books between 0-200 words: 153
            # (C1 ) Books between 200-1000 words: 1249
            # (C2 ) Books between 1000-5000 words: 11383
            # (C3 ) Books between 5000-10000 words: 16088
            # (C4 ) Books between 10000-20000 words: 20038
            # (C5 ) Books between 20000-40000 words: 23274
            # (C6 ) Books between 40000-60000 words: 15280
            # (C7 ) Books between 60000-80000 words: 11601
            # (C8 ) Books between 80000-100000 words: 8294
            # (C9 ) Books between 100000-200000 words: 11531
            # (C10) Books between 200000-300000 words: 4956
            # (C11) Books between 300000-500000 words: 3475
            # (C12) Books between 500000-1000000 words: 1807
            # (C13) Books between 1000000+ words: 712

            with open(f'dataset_len_dumps/words_sl1.pkl', 'rb') as file:
                indexes = pickle.load(file)
            markers = [0, 200, 1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 150000, 200000, 300000, 500000, 1000000]
            # fetch_count = [0, 0, 10000, 5000, 2500, 1000, 100, 50, 40, 30, 20, 10, 8, 4]  # Adjust this array for your needs
            # fetch_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            # Initialize an array to store the fetched texts
            fetched_books = []

            def normalize_newlines_regex(content):
                """
                Alternative implementation using a single regex with callback function.
                This handles all cases at once without cascading issues.
                """
                def replace_match(match):
                    newline_count = len(match.group(0))
                    
                    # Apply the rules:
                    if newline_count == 1:
                        return ' '  # 0 newlines
                    elif newline_count == 2:
                        return ' '
                    elif newline_count == 3:
                        return '\n' * 1  # 2 newlines
                    elif newline_count == 4:
                        return '\n' * 2  # 3 newlines
                    elif newline_count == 5:
                        return '\n' * 3  # 4 newlines
                    else:  # 6 or more newlines
                        return '\n' * 4  # 5 newlines (or 4 if you prefer)
                
                # Match 1 or more consecutive newlines
                return re.sub(r'\n+', replace_match, content)

            total_words_added = 0
            total_books_added = 0

            # Iterate through the markers and fetch the specified number of books
            for word_range_index, count in enumerate(fetch_count):
                if count > 0:
                    book_indices = indexes[markers[word_range_index]]
                    for i in range(min(count, len(book_indices))):
                        book_index = book_indices[i]
                        entry_data = ds['train'][book_index]['text']
                        entry_data = normalize_newlines_regex(entry_data)
                        fetched_books.append(f'{BOS_TOKEN}\n{entry_data}\n{EOS_TOKEN}{SEP_TOKEN}')
                        total_books_added += 1
                        total_words_added += len(entry_data.split())
            print(f"Added {total_books_added} books from LoC for a total of {total_words_added} additional words")

            # At this point, fetched_books contains the specified texts
            # print(f"Fetched {len(fetched_books)} books:")
            # for i, text in enumerate(fetched_books):
            #    print(f"Book {i+1} Text: [{text[10000:10600]}]...")  # Print third 100 characters

            del indexes
            del markers
            del fetch_count
            print("Extending LoC Books...")
            training_data.extend(fetched_books)
            del fetched_books
        
        # Fix text encoding
        print("Fixing text encoding...")
        index = 0
        for text in tqdm(training_data, desc="Fixing file data"):
            training_data[index] = ftfy.fix_text(text)
            index += 1

        
        # Train tokenizer
        print("Training tokenizer...")
        tokenizer = train_tokenizer(
            max_vocab_len, 
            [BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN, SEP_TOKEN], 
            training_data
        )
        
        print("Joining training data...")
        training_data = "".join(training_data)
        testing_data = "".join(testing_data)
        
        print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
        
        # Model configuration
        # MODEL_NAME = 'LoC600_d768_l16_h16_fo7_sl5'
        #MODEL_NAME = 'Cycle_Test6_fo7_sl5'
        config = ModelConfig(
            vocab_size=len(tokenizer.get_vocab()),
            max_seq_len=sequence_len,
            d_model=embed_dim,
            n_layers=num_layers,
            n_heads=num_heads,  # Increased for better parallelism
            d_ff=ffn_dim,
            dropout=weight_dropout,
            use_moe=False,
            num_experts=4,
            moe_top_k=2,
            use_flash_attn=True,
            use_rmsnorm=True,
            tie_embeddings=True,
            use_swiglu=True,
            RoPE_freq=training_rope_freq
        )
        
        # Create model
        print("Creating model...")
        model = LanguageModel(config).to(device)
        count_parameters(model)
        
        # Train
        print("Starting training...")
        trainer = OptimizedTrainer(
            model,
            tokenizer,
            config,
            device=device,
            learning_rate=maximum_lr,
            cycle_length=1000,
            weight_decay=wei_decay,
            gradient_checkpointing=False,  # Set to True if running out of memory
            compile_model=True  # Enable torch.compile for speed
        )   
        
        losses, validation_losses, testing_losses = trainer.train(
            init_start_time,
            training_data,
            testing_data,
            epochs=num_epochs,
            batch_size=num_batches,
            eval_interval=eval_iter,
            save_interval=save_iter,
            gradient_accumulation_steps=num_grad_step, # MAKE SURE BATCH_SIZE * GRADIENT STEPS IS 256 OR SMTH REASONABLE
            stride_len=stride_length,
            clean_slate_save=True,
            loading_prev=False,
            prior_name="unnamed_model",
        )
        
        # Save final model
        print("Saving final model...")
        model.save_checkpoint(f'{MODEL_NAME}', '_final', tokenizer)
        
        print(f"\nTraining complete! Total time: {(time.time() - init_start_time)/60:.2f} minutes")
