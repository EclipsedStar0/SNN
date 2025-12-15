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

PAD_TOKEN = "<PAD>"
SEP_TOKEN = "<SEP>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


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
        
        self.load_balancing_loss = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # [B*T, C]
        
        # Gating
        gates = self.gate(x_flat)  # [B*T, num_experts]
        top_k_weights, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)  # [B*T, top_k]
        
        if self.training and self.num_experts > 0:
            # Importance: how much each expert is weighted
            importance = F.softmax(gates, dim=-1).sum(0)  # [num_experts]
            
            # Load: how many tokens are routed to each expert
            load = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                load[i] = (top_k_indices == i).float().sum()
            
            # Normalize and compute loss (encourages uniform distribution)
            importance = importance / importance.sum()
            load = load / load.sum()
            self.load_balancing_loss = (importance * load).sum() * self.num_experts
        else:
            self.load_balancing_loss = None
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # For each selected expert position
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # [B*T]
            weights = top_k_weights[:, k:k+1]  # [B*T, 1]
            
            # Process each expert
            for expert_id in range(self.num_experts):
                mask = expert_indices == expert_id
                if mask.any():
                    expert_out = self.experts[expert_id](x_flat[mask])
                    output[mask] += expert_out * weights[mask]
        
        return output.view(B, T, C)
    

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
            'tokenizer_vocab': tokenizer.get_vocab(),
            #'tokenizer_merges': tokenizer.merges,
            #'tokenizer_word_freqs': tokenizer.word_freqs
        }, path)
        print(f'Checkpoint saved to {path}')
    
    @staticmethod
    def load_checkpoint(path: str, device: str = 'cpu') -> 'LanguageModel':
        """Load model from checkpoint."""
        with torch.serialization.safe_globals([ModelConfig]):
            checkpoint = torch.load(path, map_location=device)
        model = LanguageModel(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.add_tokens(checkpoint['tokenizer_vocab'].items())
        tokenizer.add_special_tokens([BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN, SEP_TOKEN])
        #tokenizer.merges = checkpoint['tokenizer_merges']
        #tokenizer.word_freqs = checkpoint['tokenizer_word_freqs']
        
        tokenizer.token_to_id = {token: idx for idx, token in enumerate(tokenizer.get_vocab())}
        tokenizer.id_to_token = {idx: token for token, idx in tokenizer.token_to_id.items()}
        
        return model.to(device), tokenizer.to(device)


class TextDataset(Dataset):
    """Dataset for language modeling."""
    
    def __init__(self, texts:str, seq_len: int, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.sequences = []
        
        print("Splitting into chunks")
        chunks = []
        initial_text_len = 0
        for chunk in texts.split(SEP_TOKEN):
            temp = chunk.strip()
            if len(temp) > 0:
                chunks.append(temp)
                initial_text_len += len(temp)
        
        chunks = [chunk.strip() for chunk in texts.split(SEP_TOKEN) if len(chunk.strip()) > 0]
        print(f"We have {len(chunks)} chunks perform sliding window on")
        
        self.sequences = []
        chunk_token_ids = []
        size_of_base_text = 0
        
        for chunk in tqdm(chunks, desc="Tokenizing chunks"):
            encoded = self.tokenizer.encode(chunk)  # This encodes the entire chunk
            size_of_base_text += len(encoded.ids)
            chunk_token_ids.append(encoded.ids)
        
        print(f"Our Corpus is {initial_text_len} characters long; Our Corpus is {size_of_base_text} tokens long.")
        print("Finding Large Chunks...")
        chunk_sizes = [len(ids) for ids in chunk_token_ids]
        large_chunks = [(i, size) for i, size in enumerate(chunk_sizes) if size > 10000]

        if large_chunks:
            print(f"\nFound {len(large_chunks)} large chunks:")
            for idx, size in large_chunks[:5]:  # Show top 5
                num_seqs = size - seq_len + 1
                print(f"  Chunk {idx}: {size:,} tokens → {num_seqs:,} sequences")
            print(f"  (Total: {sum(s for _, s in large_chunks):,} tokens in large chunks)\n")
        
        
        # Boundary-aware approach
        boundary_sequences = 0
        boundary_tokens = 0
        
        stride_len = 20
        for token_ids in tqdm(chunk_token_ids, desc="Splitting Chunks"):
            if len(token_ids) >= seq_len:
                num_sequences_in_chunk = len(token_ids) - seq_len
                boundary_sequences += num_sequences_in_chunk
                
                for i in tqdm(list(range(0, num_sequences_in_chunk + 1, stride_len)), desc="Splitting into Sequence"):
                    sequence = token_ids[i:i + seq_len + 1]
                    self.sequences.append(sequence)
                    boundary_tokens += len(sequence)  # Each sequence has seq_len+1 tokens
        
        # Calculate contiguous approach for comparison
        contiguous_sequences = max(0, (size_of_base_text - seq_len) / stride_len)
        contiguous_tokens = contiguous_sequences * (seq_len + 1)   # Each sequence has seq_len+1 tokens
        
        print(f"Boundary-aware: {boundary_sequences} sequences, {boundary_tokens} tokens")
        print(f"Contiguous: {contiguous_sequences} sequences, {contiguous_tokens} tokens")
        print(f"Reduction: {((contiguous_sequences - boundary_sequences) / contiguous_sequences * 100):.1f}% fewer sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        chunk = self.sequences[idx]
        
        # Pad if needed (should rarely happen with proper chunk filtering)
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [self.tokenizer.token_to_id(PAD_TOKEN)] * (self.seq_len + 1 - len(chunk))
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        mask = (x != self.tokenizer.token_to_id(PAD_TOKEN)).long()
        
        return x, y, mask

class MemoryEfficientTrainer:
    """Training loop with memory optimizations and mixed precision."""
    
    def __init__(self, model: LanguageModel, tokenizer: Tokenizer, config: ModelConfig,
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
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id(PAD_TOKEN))
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda')
    
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
                with torch.amp.autocast('cuda'):
                    logits = self.model(x, mask)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                
                    if self.config.use_moe and self.config.num_experts > 0:
                        lb_loss = 0.0
                        for block in self.model.blocks:
                            if hasattr(block.ffn, 'load_balancing_loss') and block.ffn.load_balancing_loss is not None:
                                lb_loss += block.ffn.load_balancing_loss
                        
                        # Add load balancing loss with a small weight (0.01 is typical)
                        loss = loss + 0.01 * lb_loss
                
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
                    self.model.save_checkpoint(f'models/text_completion_24_D_7M_guten_checkpoint_step_{global_step}.pth')
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
            encoded = self.tokenizer.encode(prompt)
            if not encoded:  # Handle empty tokenization
                continue
                
            input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(self.device)
            
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
    function_training_data = []
    directory = Path(directory_path)
    
    # For recursive search:
    number_of_files_searched = 0
    for file_path in directory.rglob("*.txt"):
        if number_of_files_searched > 600:
            break
        try:
            content = file_path.read_text(encoding='utf-8')
            function_training_data.append(f'{BOS_TOKEN}\n'+content+f'\n{EOS_TOKEN}{SEP_TOKEN}')
            print(f"Loaded: {file_path}")
            number_of_files_searched += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return function_training_data

def train_tokenizer(provided_vocab_size, provided_special_tokens, provided_training_data):
    bpe_trainer = trainers.BpeTrainer(vocab_size=provided_vocab_size, special_tokens=provided_special_tokens)
    function_tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    function_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel()
    ])
    
    function_tokenizer.decoder = decoders.ByteLevel()
    
    function_tokenizer.train_from_iterator(provided_training_data, trainer=bpe_trainer)
    return function_tokenizer

# Example usage
if __name__ == "__main__":
    # Sample data
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
        '{SEP_TOKEN}',
        f"""{BOS_TOKEN}Modern-day Terra can summarily be divided into 12 distinct bodies, of which, seven are continents, and 5 are oceans. East of the international dateline, going from the north hemisphere to the southern hemisphere before moving east, past the sprawling Pacific Ocean-- the first continent to be remarked upon is North America. Home to the United States of America, Canada, Mexico, and a sprinkling of smaller hispanic countries in Central America (the 'bridge' between North and South America), North America is considerably developed and has 618.8 million inhabitants as of November 2025. South America, the North's southern neighbour, lies to the south, beneath the equator. Home to nations such as Brazil, Columbia, and Argentine, over 439 million people live in the continent (as of 2025), of which, nearly half of the continent's population lives in Brazil. Directly across from South America, across the second largest of Terra's oceans-- the Atlantic Ocean, is Africa. A vast continent known both for its deserts and its savannahs, Africa has a long, tragic history. Exploited and ravaged by more developed locales, enslaved and pillaged by imperialists and colonial powers, despite hosting 1.53 Billion humans-- a full 18% of humanity as of 2025, Africa remains one of the most impoverished, underdeveloped continents in the world with hundreds of millions living in extreme poverty. 

Europe... Europe on the other hand, often collectively referred to as the 'West' (with the US and Canda included often times), is highly developed. The most developed continent in the world, 725.8 million people reside within Europe. For nearly half a millenia, Europe has been dominant, enslaving, pillaging, exterminating, and exploiting native populations in other continents, a trend that only began to decline in the last century, ending primarily from the World Wars and the wave of decolonization that spread across the globe. France, England, Spain, Germany, and Italy, are amongst the powers of Europe.

Across from Europe, demarked by the Baltic, Ukrainian, and Turkish border, is Asia. The most populous continent in the world with 4.97 billion people, making up nearly 60% of the global population, Asia is home to the Russian Federation, the People's Republic of China, the Republic of India, and numerous smaller nations. Interestingly enough, the Republic of India is situated on the Indian Subcontinent, a massive outcropping of land demarked by the Himalaya mountains to the north, with the nation of Nepal inhabitting the mountainous region, seperate from both China and India, amongst the two dominant great powers of Asia.

Further to the south of Asia, across the Indian Ocean, is Oceania, the sixth of the seven continents. A vast archipalego of thousands of islands big and small, Oceania only plays host to approximately 44 million inhabitants-- a paltry sum of the 8.259 billion people living on the Earth as of 2025... making up not even a single percentage point of the world's population.

And then, at the southern pole of the Earth, is Antarctica, the Earth's southernmost continent, it has the smallest population of all seven continents despite being 40% larger than Europe with a population numbering at several thousand during the summer wonths-- a number that plummets to a mere thousand come winter. On the other end of the globe, on the Earth's north pole, is the Arctic Circle a frozen collection of seas that upon which the northern tips of North America, Europe, and Asia converge upon.{EOS_TOKEN}{SEP_TOKEN}""",
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
    
    print("Extending Non-File data...")
    training_data.extend(small_training_data)
    
    index = 0
    for text in tqdm(training_data, desc="Fixing file data"):
        training_data[index] = ftfy.fix_text(text)
            
    # Train tokenizer
    print("Training tokenizer...")
    tokenizer = train_tokenizer(6000, [BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN, SEP_TOKEN], training_data)
    
    print("Joining file data...")
    training_data = "".join(training_data)
    
    
    print(tokenizer.get_vocab())
    print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
    tData = "".join(training_data)
    
    config = ModelConfig(
        vocab_size=len(tokenizer.get_vocab()),
        max_seq_len=128, # was 256
        d_model=512,
        n_layers=4, # was 6
        n_heads=2, # was 4
        dropout=0.1,
        d_ff = 2048,
        use_moe = False,  # Add this
        num_experts = 3,  # Was 4
        moe_top_k = 1    # Was 2
        # NOTE: increased dropout = increased loss; 0.3 has double loss of 0.2; 0.2 lags around 10/20 Epochs behing 0.1
        # 512 * 8 * 4 * 2048
    )
    
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
    model.save_checkpoint('models/text_completion_24_D_7M_guten_Final.pth')