from __future__ import annotations

import math
import weakref
from collections.abc import Iterator
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer
from src.techniques.cachegen import quantize_deltas_tiered

_ACTIVE_CACHEGEN_CACHE_REF: weakref.ReferenceType[CacheGenCache] | None = None


def get_active_cachegen_cache() -> CacheGenCache | None:
    """Return currently active CacheGenCache instance if alive, else None."""
    global _ACTIVE_CACHEGEN_CACHE_REF
    if _ACTIVE_CACHEGEN_CACHE_REF is not None:
        return _ACTIVE_CACHEGEN_CACHE_REF()
    return None


def set_active_cachegen_cache(cache: CacheGenCache | None) -> None:
    """Register weak reference to active CacheGenCache ensuring zero tensor retention leaks."""
    global _ACTIVE_CACHEGEN_CACHE_REF
    if cache is not None:
        _ACTIVE_CACHEGEN_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_CACHEGEN_CACHE_REF = None


class CacheGenLayer(DynamicLayer):
    """Cache layer implementing CacheGen anchor-relative deltas and layer-tiered precision."""

    is_sliding = False

    def __init__(
        self,
        layer_idx: int = 0,
        total_layers: int = 36,
        anchor_group_size: int = 10,
        shallow_bits: int = 8,
        mid_bits: int = 4,
        deep_bits: int = 2,
    ):
        self._keys_fallback: torch.Tensor | None = None
        self._values_fallback: torch.Tensor | None = None

        self.layer_idx = int(layer_idx)
        self.total_layers = int(total_layers)
        self.anchor_group_size = int(anchor_group_size)
        self.shallow_bits = int(shallow_bits)
        self.mid_bits = int(mid_bits)
        self.deep_bits = int(deep_bits)

        self.capacity: int = 0
        self.group_capacity: int = 0
        self.total_tokens_seen: int = 0
        self.head_dim: int = 0
        self.effective_bits: int = self.shallow_bits
        self.target_dtype: Any = None
        self.target_device: Any = None

        # Pre-allocated Anchor + Quantized Delta buffers
        self.k_anchors_buf: torch.Tensor | None = None
        self.k_delta_q_buf: torch.Tensor | None = None
        self.k_delta_s_buf: torch.Tensor | None = None

        self.v_anchors_buf: torch.Tensor | None = None
        self.v_delta_q_buf: torch.Tensor | None = None
        self.v_delta_s_buf: torch.Tensor | None = None

        super().__init__()

    @property
    def original_positions(self) -> list[int]:
        return list(range(self.total_tokens_seen))

    def _reconstruct_from_deltas(self, is_key: bool) -> torch.Tensor:
        t = self.total_tokens_seen
        g = self.anchor_group_size
        num_groups = math.ceil(t / g)
        if is_key:
            anchors = self.k_anchors_buf[:, :, :num_groups, :]
            dq = self.k_delta_q_buf[:, :, :t, :]
            ds = self.k_delta_s_buf[:, :, :t, :]
        else:
            anchors = self.v_anchors_buf[:, :, :num_groups, :]
            dq = self.v_delta_q_buf[:, :, :t, :]
            ds = self.v_delta_s_buf[:, :, :t, :]

        expanded_anchors = anchors.repeat_interleave(g, dim=-2)[:, :, :t, :]
        deltas_recon = dq.to(self.target_dtype) * ds.to(self.target_dtype)
        return (expanded_anchors + deltas_recon).to(self.target_dtype)

    @property
    def keys(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._keys_fallback
        return self._reconstruct_from_deltas(is_key=True)

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        self._keys_fallback = value

    @property
    def values(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._values_fallback
        return self._reconstruct_from_deltas(is_key=False)

    @values.setter
    def values(self, value: torch.Tensor | None) -> None:
        self._values_fallback = value

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.target_dtype = key_states.dtype
        self.target_device = key_states.device
        self.head_dim = key_states.shape[-1]
        self.is_initialized = True

    def _ensure_capacity(
        self,
        batch_size: int,
        num_kv: int,
        head_dim: int,
        target_device: Any,
        target_dtype: Any,
        required_capacity: int,
    ) -> None:
        if self.capacity >= required_capacity:
            return

        new_capacity = max(required_capacity, self.capacity * 2, 1024)
        new_group_cap = math.ceil(new_capacity / self.anchor_group_size)

        new_ka = torch.empty((batch_size, num_kv, new_group_cap, head_dim), dtype=target_dtype, device=target_device)
        new_kdq = torch.empty((batch_size, num_kv, new_capacity, head_dim), dtype=torch.int8, device=target_device)
        new_kds = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)

        new_va = torch.empty_like(new_ka)
        new_vdq = torch.empty_like(new_kdq)
        new_vds = torch.empty_like(new_kds)

        if self.total_tokens_seen > 0 and self.k_anchors_buf is not None:
            t = self.total_tokens_seen
            g_used = math.ceil(t / self.anchor_group_size)
            new_ka[:, :, :g_used].copy_(self.k_anchors_buf[:, :, :g_used])
            new_kdq[:, :, :t].copy_(self.k_delta_q_buf[:, :, :t])
            new_kds[:, :, :t].copy_(self.k_delta_s_buf[:, :, :t])

            new_va[:, :, :g_used].copy_(self.v_anchors_buf[:, :, :g_used])
            new_vdq[:, :, :t].copy_(self.v_delta_q_buf[:, :, :t])
            new_vds[:, :, :t].copy_(self.v_delta_s_buf[:, :, :t])

        self.k_anchors_buf = new_ka
        self.k_delta_q_buf = new_kdq
        self.k_delta_s_buf = new_kds
        self.v_anchors_buf = new_va
        self.v_delta_q_buf = new_vdq
        self.v_delta_s_buf = new_vds
        self.capacity = new_capacity
        self.group_capacity = new_group_cap

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        batch_size = key_states.shape[0]
        num_kv = key_states.shape[1]
        new_tokens = key_states.shape[-2]
        head_dim = key_states.shape[-1]

        self.target_dtype = key_states.dtype
        self.target_device = key_states.device
        self.head_dim = head_dim

        curr = self.total_tokens_seen
        self._ensure_capacity(
            batch_size, num_kv, head_dim, key_states.device, key_states.dtype, curr + new_tokens
        )

        g = self.anchor_group_size
        # Record anchors at any position where (curr + offset) % g == 0
        positions = torch.arange(curr, curr + new_tokens, device=key_states.device)
        is_anchor = (positions % g) == 0
        if is_anchor.any():
            anchor_local_idx = torch.nonzero(is_anchor, as_tuple=False).squeeze(-1)
            anchor_group_idx = positions[anchor_local_idx] // g
            self.k_anchors_buf[:, :, anchor_group_idx, :] = key_states[:, :, anchor_local_idx, :]
            self.v_anchors_buf[:, :, anchor_group_idx, :] = value_states[:, :, anchor_local_idx, :]

        # Lookup active anchors for incoming positions
        group_ids = positions // g
        active_k_anchors = self.k_anchors_buf[:, :, group_ids, :]
        active_v_anchors = self.v_anchors_buf[:, :, group_ids, :]

        k_deltas = key_states - active_k_anchors
        v_deltas = value_states - active_v_anchors

        k_q, k_s, _, bits = quantize_deltas_tiered(
            k_deltas,
            layer_idx=self.layer_idx,
            total_layers=self.total_layers,
            shallow_bits=self.shallow_bits,
            mid_bits=self.mid_bits,
            deep_bits=self.deep_bits,
        )
        v_q, v_s, _, _ = quantize_deltas_tiered(
            v_deltas,
            layer_idx=self.layer_idx,
            total_layers=self.total_layers,
            shallow_bits=self.shallow_bits,
            mid_bits=self.mid_bits,
            deep_bits=self.deep_bits,
        )
        self.effective_bits = bits

        self.k_delta_q_buf[:, :, curr : curr + new_tokens].copy_(k_q.to(torch.int8))
        self.k_delta_s_buf[:, :, curr : curr + new_tokens].copy_(k_s.to(key_states.dtype))
        self.v_delta_q_buf[:, :, curr : curr + new_tokens].copy_(v_q.to(torch.int8))
        self.v_delta_s_buf[:, :, curr : curr + new_tokens].copy_(v_s.to(value_states.dtype))

        self.total_tokens_seen += new_tokens

        recon_k = self.keys
        recon_v = self.values
        assert recon_k is not None and recon_v is not None
        return recon_k, recon_v

    @property
    def compressed_bytes(self) -> int:
        if self.total_tokens_seen == 0 or self.k_anchors_buf is None:
            return 0
        t = self.total_tokens_seen
        g_used = math.ceil(t / self.anchor_group_size)
        batch, heads = self.k_delta_q_buf.shape[0], self.k_delta_q_buf.shape[1]
        anchor_bytes = 2 * batch * heads * g_used * self.head_dim * self.k_anchors_buf.element_size()
        # Bitstream delta footprint governed by layer-tiered bitwidth
        delta_bits_bytes = int(math.ceil(2 * batch * heads * t * self.head_dim * (self.effective_bits / 8.0)))
        scale_bytes = 2 * batch * heads * t * self.k_delta_s_buf.element_size()
        return anchor_bytes + delta_bits_bytes + scale_bytes

    def get_seq_length(self) -> int:
        return self.total_tokens_seen

    def get_max_length(self) -> int:
        return max(self.total_tokens_seen, 32768)

    def reset(self) -> None:
        self.total_tokens_seen = 0
        self.is_initialized = False
        self._keys_fallback = None
        self._values_fallback = None
        if hasattr(self, "cumulative_length"):
            if isinstance(self.cumulative_length, int):
                self.cumulative_length = 0
            else:
                self.cumulative_length.zero_()


class CacheGenCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing CacheGen anchor-delta storage."""

    is_cachegen = True

    def __init__(
        self,
        total_layers: int = 36,
        anchor_group_size: int = 10,
        shallow_bits: int = 8,
        mid_bits: int = 4,
        deep_bits: int = 2,
    ):
        self.total_layers = int(total_layers)
        self.anchor_group_size = int(anchor_group_size)
        self.shallow_bits = int(shallow_bits)
        self.mid_bits = int(mid_bits)
        self.deep_bits = int(deep_bits)

        super().__init__(
            layer_class_to_replicate=lambda: CacheGenLayer(
                layer_idx=len(self.layers),
                total_layers=self.total_layers,
                anchor_group_size=self.anchor_group_size,
                shallow_bits=self.shallow_bits,
                mid_bits=self.mid_bits,
                deep_bits=self.deep_bits,
            )
        )
        set_active_cachegen_cache(self)

    @property
    def original_positions(self) -> list[int]:
        if not self.layers:
            return []
        return self.layers[0].original_positions

    @property
    def total_compressed_bytes(self) -> int:
        return sum(getattr(layer, "compressed_bytes", 0) for layer in self.layers)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self) -> int:
        if not self.layers:
            return 32768
        return self.layers[0].get_max_length()

    def __iter__(self) -> Iterator[tuple[Any, Any, Any]]:
        for layer in self.layers:
            yield getattr(layer, "keys", None), getattr(layer, "values", None), None
