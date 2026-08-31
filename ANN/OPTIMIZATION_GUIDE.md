# Language Model Optimization Guide

This document explains the key optimizations applied from the state-of-the-art "optimized" model to your custom implementation.

## Overview

The optimized model incorporates 9 major categories of improvements that can lead to **2-5x faster training** and **better model quality** without requiring custom kernels or specialized hardware.

---

## 1. RMSNorm Instead of LayerNorm

**What Changed:**
- Replaced `nn.LayerNorm` with `RMSNorm` (Root Mean Square Layer Normalization)

**Why It Matters:**
- RMSNorm is 10-20% faster than LayerNorm
- Removes the mean centering operation (only normalizes by RMS)
- Used in modern models like LLaMA

**Implementation:**
```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
```

---

## 2. Improved Rotary Position Embeddings (RoPE)

**What Changed:**
- Added efficient caching mechanism for cos/sin values
- Used JIT compilation for the apply_rotary_emb function
- Optimized the rotation computation

**Why It Matters:**
- Avoids recomputing the same cos/sin values repeatedly
- JIT compilation makes rotation ~2x faster
- Better than learned positional embeddings for longer sequences

**Key Improvements:**
```python
@torch.jit.script
def apply_rotary_emb(q, k, cos, sin):
    # JIT compiled for speed
    # Efficient split and rotation
```

---

## 3. Flash Attention Support

**What Changed:**
- Added support for `F.scaled_dot_product_attention` (Flash Attention)
- Falls back to manual implementation if not available

**Why It Matters:**
- Flash Attention is **3-5x faster** than manual attention
- Uses less memory through kernel fusion
- Available in PyTorch 2.0+

**Configuration:**
```python
if self.flash and hasattr(F, 'scaled_dot_product_attention'):
    attn_output = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=True,
        scale=self.scale
    )
```

---

## 4. QK Normalization

**What Changed:**
- Added normalization to queries and keys before attention

**Why It Matters:**
- Improves training stability
- Reduces need for careful learning rate tuning
- Used in modern architectures

**Implementation:**
```python
# After reshaping q and k for multi-head attention
q = F.normalize(q, p=2, dim=-1)
k = F.normalize(k, p=2, dim=-1)
```

---

## 5. SwiGLU Activation Function

**What Changed:**
- Replaced GELU/ReLU in FFN with SwiGLU
- Adjusted hidden dimension accordingly

**Why It Matters:**
- SwiGLU performs better than standard activations
- Used in LLaMA, PaLM, and other SOTA models
- Adds gating mechanism to the feed-forward network

**Implementation:**
```python
class SwiGLU(nn.Module):
    def forward(self, x):
        # SwiGLU: Swish(W1*x) ⊗ (W2*x)
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))
```

---

## 6. Zero Initialization for Output Projections

**What Changed:**
- Output projections in attention and FFN are initialized to zero
- Residual branch scaling factors are learned parameters

**Why It Matters:**
- Improves training stability at initialization
- Each layer starts as an identity function
- Allows training very deep networks

**Implementation:**
```python
# Zero init for output projections
nn.init.zeros_(self.out_proj.weight)
nn.init.zeros_(self.w3.weight)

# Learnable residual scaling
self.alpha_attn = nn.Parameter(torch.ones(1))
self.alpha_ffn = nn.Parameter(torch.ones(1))

# In forward:
x = x + self.alpha_attn * self.attn(...)
x = x + self.alpha_ffn * self.ffn(...)
```

---

## 7. Weight Tying (Embedding-LM Head)

**What Changed:**
- Tied input embedding weights with output LM head weights

**Why It Matters:**
- Reduces parameters by vocab_size × d_model (~12M parameters for vocab=24k, d=512)
- Improves generalization
- Standard practice in modern LMs

**Implementation:**
```python
if config.tie_embeddings:
    self.token_embedding.weight = self.lm_head.weight
```

---

## 8. Improved Weight Initialization

**What Changed:**
- Smaller initialization std (0.02 instead of default)
- Scaled initialization for residual projections based on depth
- Follows GPT-2/GPT-3 initialization scheme

**Why It Matters:**
- Better training stability
- Faster convergence
- Critical for deep networks

**Implementation:**
```python
# Standard init
nn.init.normal_(module.weight, mean=0.0, std=0.02)

# Scaled init for residual projections
for pn, p in self.named_parameters():
    if pn.endswith('out_proj.weight') or pn.endswith('w3.weight'):
        nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))
```

---

## 9. Training Optimizations

### a) Fused AdamW
**What Changed:**
- Use `fused=True` in AdamW optimizer

**Why It Matters:**
- 10-20% faster optimizer step
- Available on CUDA devices

```python
self.optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(0.9, 0.95),  # Beta2=0.95 is better for language models
    eps=1e-8,
    fused=True  # Fused kernel
)
```

### b) Mixed Precision Training
**What Changed:**
- Use `torch.amp.autocast` and `GradScaler`

**Why It Matters:**
- **2x faster training** on modern GPUs
- Reduces memory usage by ~40%
- Maintains model quality

```python
with torch.amp.autocast('cuda', enabled=True):
    logits = self.model(data_input, mask)
    loss = self.criterion(logits.view(-1, logits.size(-1)), data_output.view(-1))

self.scaler.scale(loss).backward()
```

### c) Gradient Accumulation
**What Changed:**
- Added explicit gradient accumulation support

**Why It Matters:**
- Simulate larger batch sizes without OOM
- Better gradient estimates
- More stable training

```python
loss = loss / gradient_accumulation_steps
loss.backward()

if (batch_idx + 1) % gradient_accumulation_steps == 0:
    # Clip and step
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
```

### d) Cosine Learning Rate Schedule with Warmup
**What Changed:**
- Linear warmup followed by cosine decay

**Why It Matters:**
- Better convergence than fixed LR
- Standard in modern LM training

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
```

### e) Better Data Loading
**What Changed:**
- `num_workers=2` for parallel loading
- `pin_memory=True` for faster GPU transfer
- `persistent_workers=True` to avoid worker restart overhead

**Why It Matters:**
- Reduces data loading bottleneck
- GPU stays busier

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)
```

### f) Torch Compile (PyTorch 2.0+)
**What Changed:**
- Wrap model in `torch.compile()`

**Why It Matters:**
- **20-50% speedup** with no code changes
- Graph optimization and kernel fusion

```python
if compile_model and hasattr(torch, 'compile'):
    self.model = torch.compile(self.model)
```

---

## 10. Better Dataset Construction

**What Changed:**
- Use overlapping windows with stride
- More efficient tokenization

**Why It Matters:**
- Better data efficiency (more training examples)
- Each token appears in multiple contexts

```python
stride_len = seq_len // 2  # 50% overlap
for i in range(0, len(token_ids) - seq_len, stride_len):
    sequence = token_ids[i:i + seq_len + 1]
    self.sequences.append(sequence)
```

---

## 11. Improved MoE Load Balancing

**What Changed:**
- Better load balancing loss calculation
- Auxiliary loss encourages uniform expert usage

**Why It Matters:**
- Prevents expert collapse (all tokens routed to one expert)
- Better utilization of model capacity

```python
importance = F.softmax(gates, dim=-1).sum(0)
load = torch.zeros(self.num_experts, device=x.device)
for i in range(self.num_experts):
    load[i] = (top_k_indices == i).float().sum()

importance = importance / importance.sum()
load = load / (load.sum() + 1e-10)
self.load_balancing_loss = (importance * load).sum() * self.num_experts
```

---

## Expected Performance Improvements

### Training Speed
- **2-5x faster** depending on GPU and model size
- Key contributors:
  - Flash Attention: 3-5x
  - Mixed Precision: 2x
  - Fused AdamW: 1.2x
  - RMSNorm: 1.1x
  - Torch Compile: 1.3x
  - **Combined effect is multiplicative!**

### Memory Usage
- **30-40% less memory** with mixed precision
- Enables training larger models or bigger batches

### Model Quality
- Better convergence from improved initialization
- More stable training from QK normalization
- Better performance from SwiGLU and weight tying

---

## Configuration Recommendations

### For Small Models (< 100M parameters)
```python
config = ModelConfig(
    vocab_size=24000,
    max_seq_len=256,
    d_model=512,
    n_layers=12,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
    use_flash_attn=True,
    use_rmsnorm=True,
    tie_embeddings=True,
    use_swiglu=True
)

trainer = OptimizedTrainer(
    model, tokenizer, config,
    learning_rate=3e-4,
    weight_decay=0.1,
    gradient_checkpointing=False,
    compile_model=True
)
```

### For Medium Models (100M - 1B parameters)
```python
config = ModelConfig(
    vocab_size=32000,
    max_seq_len=512,
    d_model=1024,
    n_layers=24,
    n_heads=16,
    d_ff=4096,
    dropout=0.1,
    use_flash_attn=True,
    use_rmsnorm=True,
    tie_embeddings=True,
    use_swiglu=True
)

trainer = OptimizedTrainer(
    model, tokenizer, config,
    learning_rate=2e-4,
    weight_decay=0.1,
    gradient_checkpointing=True,  # Enable for memory
    compile_model=True
)
```

---

## What Was NOT Included (From Optimized Code)

These require custom kernels or specialized environments:

1. **FP8 Quantization** - Requires A100/H100 GPUs and custom CUDA kernels
2. **Custom Triton Kernels** - Platform-specific, harder to deploy
3. **Distributed Training** - DDP/FSDP requires multi-GPU setup
4. **Custom Optimizer (NorMuon)** - Complex implementation, AdamW works great
5. **Extensive Hyperparameter Scheduling** - Simplified for clarity

---

## Quick Migration Guide

### Step 1: Replace Normalization
```python
# Old
self.ln1 = nn.LayerNorm(config.d_model)

# New
self.ln1 = RMSNorm(config.d_model)
```

### Step 2: Update Attention
```python
# Add to __init__:
self.flash = config.use_flash_attn and hasattr(F, 'scaled_dot_product_attention')

# In forward, replace manual attention with:
if self.flash:
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### Step 3: Switch to SwiGLU
```python
# Replace FeedForward with SwiGLU
self.ffn = SwiGLU(config)
```

### Step 4: Enable Weight Tying
```python
# In model __init__:
if config.tie_embeddings:
    self.token_embedding.weight = self.lm_head.weight
```

### Step 5: Update Training Loop
```python
# Enable mixed precision
with torch.amp.autocast('cuda'):
    logits = model(x)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Step 6: Use Fused Optimizer
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    fused=True  # Add this
)
```

### Step 7: Compile Model
```python
model = torch.compile(model)
```

---

## Testing Your Optimizations

### Benchmark Speed
```python
import time

# Before optimizations
start = time.time()
for _ in range(100):
    loss = model(x, y)
    loss.backward()
print(f"Old: {time.time() - start:.2f}s")

# After optimizations
start = time.time()
for _ in range(100):
    with torch.amp.autocast('cuda'):
        loss = optimized_model(x, y)
    scaler.scale(loss).backward()
print(f"New: {time.time() - start:.2f}s")
```

### Check Memory Usage
```python
import torch

torch.cuda.reset_peak_memory_stats()
# Run training step
peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_memory:.2f} GB")
```

---

## Troubleshooting

### "CUDA out of memory"
- Enable gradient checkpointing: `gradient_checkpointing=True`
- Reduce batch size
- Reduce sequence length
- Enable gradient accumulation

### "Compile not working"
- Requires PyTorch 2.0+
- Some operations may not be compile-compatible
- Try `torch.compile(model, mode="reduce-overhead")` for better compatibility

### "Flash attention not available"
- Requires PyTorch 2.0+
- Falls back to manual attention automatically
- Check with: `hasattr(F, 'scaled_dot_product_attention')`

### "Training unstable"
- Reduce learning rate
- Increase warmup steps
- Check gradient clipping is enabled
- Verify initialization is correct

---

## Summary

The optimized model incorporates modern best practices that provide:
- ✅ **2-5x faster training**
- ✅ **30-40% less memory**
- ✅ **Better model quality**
- ✅ **More stable training**
- ✅ **No custom kernels required**
- ✅ **Works on any CUDA GPU**

All optimizations are production-ready and widely used in state-of-the-art language models.
