from __future__ import annotations

import weakref
from collections.abc import Iterator
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer
from src.techniques.h2o import compute_h2o_boundaries, evict_h2o_kv_cache

_ACTIVE_H2O_CACHE_REF: weakref.ReferenceType[H2OCache] | None = None


def get_active_h2o_cache() -> H2OCache | None:
    """Return currently active H2OCache instance if alive, else None."""
    global _ACTIVE_H2O_CACHE_REF
    if _ACTIVE_H2O_CACHE_REF is not None:
        return _ACTIVE_H2O_CACHE_REF()
    return None


def set_active_h2o_cache(cache: H2OCache | None) -> None:
    """Register weak reference to active H2OCache ensuring zero tensor retention leaks."""
    global _ACTIVE_H2O_CACHE_REF
    if cache is not None:
        _ACTIVE_H2O_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_H2O_CACHE_REF = None


def record_h2o_attention_scores(
    layer: H2OLayer,
    query: torch.Tensor,
    key: torch.Tensor,
) -> None:
    """Compute step or chunk attention scores, pool across GQA heads, and accumulate into layer.

    Supports both single-token decode (q_len == 1, vector-matrix multiply) and
    chunked prefill (q_len > 1 with causal masking).
    """
    if torch is None or F is None:
        return

    batch = query.shape[0]
    num_q_heads = query.shape[1]
    q_len = query.shape[2]
    head_dim = query.shape[3]
    num_kv_heads = key.shape[1]
    kv_len = key.shape[2]

    if num_q_heads % num_kv_heads != 0:
        return
    group_size = num_q_heads // num_kv_heads

    if q_len == 1:
        # Fast decode path: vector-matrix dot-product (< 5 us, minimal memory)
        q_grouped = query.view(batch, num_kv_heads, group_size, 1, head_dim)
        k_expanded = key.unsqueeze(2)  # (batch, num_kv_heads, 1, kv_len, head_dim)
        scores = (q_grouped.float() * k_expanded.float()).sum(dim=-1) * (head_dim ** -0.5)
        attn = F.softmax(scores, dim=-1)
        step_scores = attn.mean(dim=2).squeeze(2)  # (batch, num_kv_heads, kv_len)
        layer.record_scores(step_scores)
    else:
        # Chunked prefill path: causal chunk attention matrix
        k_rep = key.repeat_interleave(group_size, dim=1)
        scores = torch.matmul(query.float(), k_rep.float().transpose(-2, -1)) * (head_dim ** -0.5)
        past_len = kv_len - q_len
        q_idx = torch.arange(q_len, device=query.device).unsqueeze(-1) + past_len
        kv_idx = torch.arange(kv_len, device=query.device).unsqueeze(0)
        mask = kv_idx <= q_idx
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn_grouped = attn.view(batch, num_kv_heads, group_size, q_len, kv_len)
        chunk_scores = attn_grouped.mean(dim=2).sum(dim=2)  # (batch, num_kv_heads, kv_len)
        layer.record_scores(chunk_scores)


class H2OLayer(DynamicLayer):
    """Cache layer implementing exact Heavy-Hitter Oracle (H2O) cumulative score tracking and eviction."""

    is_sliding = False

    def __init__(self, heavy_hitter_tokens: int = 1024, recent_window_tokens: int = 1024):
        super().__init__()
        if int(heavy_hitter_tokens) < 0:
            raise ValueError(f"heavy_hitter_tokens must be >= 0, got {heavy_hitter_tokens}")
        if int(recent_window_tokens) <= 0:
            raise ValueError(f"recent_window_tokens must be > 0, got {recent_window_tokens}")

        self.heavy_hitter_tokens = int(heavy_hitter_tokens)
        self.recent_window_tokens = int(recent_window_tokens)
        self.cumulative_scores: torch.Tensor | None = None
        self.original_positions: list[int] = []
        self.total_tokens_seen: int = 0

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append incoming key and value states, expanding cumulative score tracking tensor."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

        new_tokens = key_states.shape[-2]
        self.original_positions.extend(range(self.total_tokens_seen, self.total_tokens_seen + new_tokens))
        self.total_tokens_seen += new_tokens
        batch = key_states.shape[0]
        num_kv_heads = key_states.shape[1]

        if self.cumulative_scores is None:
            self.cumulative_scores = torch.zeros(
                batch, num_kv_heads, new_tokens, device=key_states.device, dtype=torch.float32
            )
        else:
            zeros = torch.zeros(
                batch, num_kv_heads, new_tokens, device=key_states.device, dtype=torch.float32
            )
            self.cumulative_scores = torch.cat([self.cumulative_scores, zeros], dim=-1)

        return self.keys, self.values

    def record_scores(self, step_scores: torch.Tensor) -> None:
        """Accumulate step attention scores into the layer's cumulative score tensor."""
        if self.cumulative_scores is not None:
            # Align lengths if needed
            scores_aligned = step_scores.to(device=self.cumulative_scores.device, dtype=torch.float32)
            if scores_aligned.shape == self.cumulative_scores.shape:
                self.cumulative_scores += scores_aligned
            elif scores_aligned.shape[-1] <= self.cumulative_scores.shape[-1]:
                self.cumulative_scores[:, :, : scores_aligned.shape[-1]] += scores_aligned
            else:
                self.cumulative_scores += scores_aligned[:, :, : self.cumulative_scores.shape[-1]]

    def prune(self) -> None:
        """Prune low-attention candidate tokens outside recent window if over capacity."""
        if self.is_initialized and self.keys is not None and self.values is not None:
            seq_len = self.keys.shape[-2]
            capacity = self.heavy_hitter_tokens + self.recent_window_tokens
            if seq_len > capacity:
                boundary = compute_h2o_boundaries(
                    seq_len=seq_len,
                    cumulative_scores=self.cumulative_scores,
                    heavy_hitter_tokens=self.heavy_hitter_tokens,
                    recent_window_tokens=self.recent_window_tokens,
                )
                self.keys, self.values = evict_h2o_kv_cache(
                    self.keys,
                    self.values,
                    retained_indices=boundary.retained_indices,
                    seq_dim=-2,
                )
                if self.original_positions and len(self.original_positions) >= len(boundary.retained_indices):
                    self.original_positions = [
                        self.original_positions[i]
                        for i in boundary.retained_indices
                        if i < len(self.original_positions)
                    ]
                if self.cumulative_scores is not None:
                    idx_tensor = torch.tensor(
                        boundary.retained_indices, device=self.cumulative_scores.device, dtype=torch.long
                    )
                    self.cumulative_scores = torch.index_select(
                        self.cumulative_scores, dim=-1, index=idx_tensor
                    )

    def get_max_length(self) -> int:
        return self.heavy_hitter_tokens + self.recent_window_tokens

    def get_seq_length(self) -> int:
        return self.keys.shape[-2] if (self.is_initialized and self.keys is not None) else 0


class H2OCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing exact H2O retention.

    Dynamically maintains top-$k_{hh}$ heavy hitters with highest cumulative attention
    alongside a rolling recent window of $k_{recent}$ tokens across arbitrary sequence lengths.
    """

    is_h2o = True

    def __init__(
        self,
        heavy_hitter_tokens: int = 1024,
        recent_window_tokens: int = 1024,
    ):
        if int(heavy_hitter_tokens) < 0:
            raise ValueError(f"heavy_hitter_tokens must be >= 0, got {heavy_hitter_tokens}")
        if int(recent_window_tokens) <= 0:
            raise ValueError(f"recent_window_tokens must be > 0, got {recent_window_tokens}")

        self.heavy_hitter_tokens = int(heavy_hitter_tokens)
        self.recent_window_tokens = int(recent_window_tokens)

        super().__init__(
            layer_class_to_replicate=lambda: H2OLayer(
                heavy_hitter_tokens=self.heavy_hitter_tokens,
                recent_window_tokens=self.recent_window_tokens,
            )
        )
        set_active_h2o_cache(self)

    def prune_after_chunk(self) -> None:
        """Evict middle tokens across all initialized layers after prefill chunk processing."""
        for layer in self.layers:
            if hasattr(layer, "prune"):
                layer.prune()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self) -> int:
        return self.heavy_hitter_tokens + self.recent_window_tokens

    def __iter__(self) -> Iterator[tuple[Any, Any, Any]]:
        for layer in self.layers:
            yield getattr(layer, "keys", None), getattr(layer, "values", None), None
