"""
qat.py — Quantization-Aware Training (QAT) for LanguageModel
=============================================================

Drop this file next to your main training script and import from it.

Usage
-----
from qat import (
    QATConfig,
    wrap_model_for_qat,
    maybe_enable_qat,
    export_quantized_model,
    QATOptimizedTrainer,
)

Quick-start inside your __main__ block
---------------------------------------
    qat_cfg = QATConfig(qat_start_step=2000, bits=8)
    model    = wrap_model_for_qat(model, qat_cfg)          # swap Linear → QATLinear
    trainer  = QATOptimizedTrainer(model, tokenizer, config,
                                   qat_config=qat_cfg, ...)
    # training proceeds normally; QAT is activated at step qat_start_step
    # after training:
    int8_model = export_quantized_model(model, qat_cfg, save_path="models/mymodel/mymodel_int8.pt")
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# QAT Configuration
# ---------------------------------------------------------------------------

@dataclass
class QATConfig:
    """All knobs for quantization-aware training."""

    # ── Bit-width ────────────────────────────────────────────────────────────
    bits: int = 8                    # quantization bit-width (4 or 8 recommended)

    # ── Activation quantization ──────────────────────────────────────────────
    quantize_activations: bool = True
    act_quant_per_tensor: bool = True   # per-tensor (True) or per-channel (False)

    # ── Weight quantization ──────────────────────────────────────────────────
    quantize_weights: bool = True
    weight_quant_per_channel: bool = True  # per-channel is higher quality

    # ── Schedule ─────────────────────────────────────────────────────────────
    qat_start_step: int = 1000       # global optimiser step at which QAT is switched on
    # Exponential-Moving-Average for observer scale/zero-point
    ema_momentum: float = 0.1

    # ── Layers to skip ───────────────────────────────────────────────────────
    # Module name substrings that should NOT be quantized (e.g. final head)
    skip_modules: list[str] = field(default_factory=lambda: ["lm_head"])


# ---------------------------------------------------------------------------
# Helpers: quantisation math
# ---------------------------------------------------------------------------

def _compute_qparams(x: torch.Tensor, bits: int, per_channel: bool, ch_dim: int = 0):
    """
    Compute scale and zero_point for symmetric INT quantization.

    Returns
    -------
    scale     : same shape as x if per_channel else scalar tensor
    zero_point: always 0 for symmetric quant (returned for API completeness)
    """
    qmax = (2 ** (bits - 1)) - 1  # e.g. 127 for INT8

    if per_channel:
        # Reduce over every dimension *except* ch_dim
        dims = list(range(x.dim()))
        dims.pop(ch_dim)
        abs_max = x.abs().amax(dim=dims, keepdim=True).clamp(min=1e-8)
    else:
        abs_max = x.abs().amax().clamp(min=1e-8)

    scale = abs_max / qmax
    zero_point = torch.zeros_like(scale, dtype=torch.long)
    return scale, zero_point


@torch.jit.script
def _fake_quantize_sym(x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric fake-quantize with straight-through estimator (STE)."""
    qmax = float((2 ** (bits - 1)) - 1)
    x_q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    # STE: gradient passes through the clamp but not the round
    x_dq = x_q * scale
    return x + (x_dq - x).detach()          # STE trick


# ---------------------------------------------------------------------------
# FakeQuantize observer (EMA-based)
# ---------------------------------------------------------------------------

class EMAObserver(nn.Module):
    """Tracks running abs-max via EMA; emits scale for fake-quantize."""

    def __init__(self, bits: int, per_channel: bool, ch_dim: int, momentum: float):
        super().__init__()
        self.bits = bits
        self.per_channel = per_channel
        self.ch_dim = ch_dim
        self.momentum = momentum
        self.register_buffer("_ema_max", torch.tensor(1.0))
        self._initialized = False

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> torch.Tensor:
        """Update EMA and return current scale."""
        if self.per_channel:
            dims = list(range(x.dim()))
            dims.pop(self.ch_dim)
            cur_max = x.abs().amax(dim=dims).view(-1)
        else:
            cur_max = x.abs().amax().unsqueeze(0)

        if not self._initialized or self._ema_max.shape != cur_max.shape:
            self._ema_max = cur_max.clone()
            self._initialized = True
        else:
            self._ema_max = (1 - self.momentum) * self._ema_max + self.momentum * cur_max

        qmax = (2 ** (self.bits - 1)) - 1
        scale = (self._ema_max / qmax).clamp(min=1e-8)
        return scale


# ---------------------------------------------------------------------------
# QATLinear — drop-in replacement for nn.Linear
# ---------------------------------------------------------------------------

class QATLinear(nn.Linear):
    """
    nn.Linear with optional fake-quantization of weights and activations.

    The layer is *transparent* when `qat_active = False` — identical to
    standard nn.Linear.  Call `.enable_qat()` / `.disable_qat()` at any time.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qat_cfg: Optional[QATConfig] = None,
    ):
        super().__init__(in_features, out_features, bias)
        self.qat_cfg = qat_cfg or QATConfig()
        self.qat_active = False

        cfg = self.qat_cfg

        # Weight observer (per-channel on output dim)
        if cfg.quantize_weights:
            self.weight_observer = EMAObserver(
                bits=cfg.bits,
                per_channel=cfg.weight_quant_per_channel,
                ch_dim=0,
                momentum=cfg.ema_momentum,
            )

        # Activation observer (per-tensor on input to this layer)
        if cfg.quantize_activations:
            self.act_observer = EMAObserver(
                bits=cfg.bits,
                per_channel=not cfg.act_quant_per_tensor,
                ch_dim=-1,
                momentum=cfg.ema_momentum,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def enable_qat(self):
        self.qat_active = True

    def disable_qat(self):
        self.qat_active = False

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.qat_active:
            return super().forward(x)

        cfg = self.qat_cfg

        # -- Fake-quantize activations (input) --------------------------------
        if cfg.quantize_activations:
            act_scale = self.act_observer.update(x)
            # broadcast shape: scalar or (C,) along last dim
            if act_scale.numel() > 1:
                act_scale = act_scale.view(*([1] * (x.dim() - 1)), -1)
            x = _fake_quantize_sym(x, act_scale, cfg.bits)

        # -- Fake-quantize weights --------------------------------------------
        w = self.weight
        if cfg.quantize_weights:
            w_scale = self.weight_observer.update(w)      # (out_features,) or scalar
            if w_scale.numel() > 1:
                # per-channel: shape (out_features, 1, …)
                w_scale = w_scale.view(-1, *([1] * (w.dim() - 1)))
            w = _fake_quantize_sym(w, w_scale, cfg.bits)

        return F.linear(x, w, self.bias)

    # ── Serialisation ─────────────────────────────────────────────────────────

    @classmethod
    def from_linear(cls, linear: nn.Linear, qat_cfg: QATConfig) -> "QATLinear":
        """Convert an existing nn.Linear in-place (shares weight tensor)."""
        new = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            qat_cfg=qat_cfg,
        )
        new.weight = linear.weight          # shared reference
        if linear.bias is not None:
            new.bias = linear.bias
        return new

    def to_real_quantized(self) -> nn.Linear:
        """
        Export to a plain nn.Linear whose weights are rounded to INT and
        dequantized — useful for final export / inference.
        (True INT8 inference requires a dedicated runtime; this gives you
        the rounded weights for inspection or TorchScript export.)
        """
        cfg = self.qat_cfg
        with torch.no_grad():
            w = self.weight.float()
            if cfg.quantize_weights:
                scale = self.weight_observer.update(w)
                if scale.numel() > 1:
                    scale = scale.view(-1, *([1] * (w.dim() - 1)))
                qmax = (2 ** (cfg.bits - 1)) - 1
                w = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale

        out = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        out.weight = nn.Parameter(w.to(self.weight.dtype))
        if self.bias is not None:
            out.bias = nn.Parameter(self.bias.data.clone())
        return out


# ---------------------------------------------------------------------------
# Model surgery: swap nn.Linear → QATLinear
# ---------------------------------------------------------------------------

def wrap_model_for_qat(model: nn.Module, qat_cfg: QATConfig) -> nn.Module:
    """
    Recursively replace every nn.Linear in *model* with a QATLinear.
    Modules whose fully-qualified name contains any string in
    ``qat_cfg.skip_modules`` are left untouched.

    The replacement shares the original weight tensor, so no extra memory
    is allocated and any existing optimizer references remain valid.

    Returns the (mutated) model.
    """
    def _replace(parent: nn.Module, prefix: str = ""):
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            # Skip explicitly excluded modules
            if any(skip in full_name for skip in qat_cfg.skip_modules):
                continue

            if isinstance(child, nn.Linear) and not isinstance(child, QATLinear):
                setattr(parent, name, QATLinear.from_linear(child, qat_cfg))
            else:
                _replace(child, full_name)

    _replace(model)
    print(
        f"[QAT] Replaced nn.Linear layers with QATLinear "
        f"(bits={qat_cfg.bits}, skip={qat_cfg.skip_modules})."
    )
    return model


# ---------------------------------------------------------------------------
# Convenience functions to toggle QAT across entire model
# ---------------------------------------------------------------------------

def enable_qat(model: nn.Module):
    """Activate fake-quantization in every QATLinear inside *model*."""
    n = 0
    for m in model.modules():
        if isinstance(m, QATLinear):
            m.enable_qat()
            n += 1
    print(f"[QAT] Enabled QAT in {n} QATLinear layers.")


def disable_qat(model: nn.Module):
    """Deactivate fake-quantization (model behaves as float32)."""
    n = 0
    for m in model.modules():
        if isinstance(m, QATLinear):
            m.disable_qat()
            n += 1
    print(f"[QAT] Disabled QAT in {n} QATLinear layers.")


def maybe_enable_qat(model: nn.Module, qat_cfg: QATConfig, global_step: int) -> bool:
    """
    Call this once per optimiser step.  Returns True the *first* time QAT
    is switched on (so you can log it).
    """
    if global_step == qat_cfg.qat_start_step:
        enable_qat(model)
        return True
    return False


# ---------------------------------------------------------------------------
# Export: convert QATLinear → plain INT-rounded Linear for deployment
# ---------------------------------------------------------------------------

def export_quantized_model(
    model: nn.Module,
    qat_cfg: QATConfig,
    save_path: Optional[str] = None,
) -> nn.Module:
    """
    Walk *model*, convert every QATLinear to a dequantized nn.Linear whose
    weights are rounded to the nearest representable INT value, then
    optionally save with torch.save.

    Returns the converted model (original is NOT mutated).
    """
    import copy
    exported = copy.deepcopy(model)
    exported.eval()

    def _convert(parent: nn.Module):
        for name, child in list(parent.named_children()):
            if isinstance(child, QATLinear):
                setattr(parent, name, child.to_real_quantized())
            else:
                _convert(child)

    _convert(exported)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(exported.state_dict(), save_path)
        print(f"[QAT] Exported quantized model state dict → {save_path}")

    return exported


# ---------------------------------------------------------------------------
# QATOptimizedTrainer — thin subclass that adds QAT scheduling
# ---------------------------------------------------------------------------

# We import the base trainer lazily to avoid circular imports if this file
# is used standalone.  If your main file is called `train.py`, adjust the
# import below.

try:
    # Attempt import from parent module (adjust name to match your file)
    import importlib, sys
    _parent = importlib.import_module("__main__")
    _OptimizedTrainer = getattr(_parent, "OptimizedTrainer", None)
except Exception:
    _OptimizedTrainer = None


class QATOptimizedTrainer:
    """
    Wraps OptimizedTrainer and injects QAT activation at the right step.

    If OptimizedTrainer is not importable (e.g. standalone usage), this
    class provides a minimal fallback that demonstrates the hook pattern.
    You can also simply copy the ``_qat_step_hook`` logic directly into
    your existing ``OptimizedTrainer.train`` loop.
    """

    def __init__(self, model, tokenizer, config, qat_config: QATConfig, **trainer_kwargs):
        self.qat_config = qat_config
        self._qat_enabled = False

        if _OptimizedTrainer is not None:
            self._inner = _OptimizedTrainer(model, tokenizer, config, **trainer_kwargs)
            # Monkey-patch the inner loop step callback
            self._inner._qat_step_hook = self._qat_step_hook
        else:
            raise RuntimeError(
                "OptimizedTrainer not found.  Either import QATOptimizedTrainer from "
                "within your training script, or copy `_qat_step_hook` manually."
            )

    # ── Hook called inside the training loop ─────────────────────────────────

    def _qat_step_hook(self, global_step: int):
        """Insert this call at the top of the gradient-accumulation block."""
        if not self._qat_enabled and global_step >= self.qat_config.qat_start_step:
            enable_qat(self._inner.model)
            self._qat_enabled = True
            print(
                f"\n[QAT] ★ Quantization-Aware Training ACTIVATED at step {global_step} "
                f"(bits={self.qat_config.bits}) ★\n"
            )

    # ── Delegate everything else to the inner trainer ─────────────────────────

    def train(self, *args, **kwargs):
        return self._inner.train(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)