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
        self.effective_bits: int = self._resolve_tier_bits()
        self.pack_bits: int = 8
        self.target_dtype: Any = None
        self.target_device: Any = None

        # Pre-allocated Anchor + Bit-Packed Quantized Delta buffers
        self.k_anchors_buf: torch.Tensor | None = None
        self.k_delta_q_buf: torch.Tensor | None = None
        self.k_delta_s_buf: torch.Tensor | None = None

        self.v_anchors_buf: torch.Tensor | None = None
        self.v_delta_q_buf: torch.Tensor | None = None
        self.v_delta_s_buf: torch.Tensor | None = None

        super().__init__()

    def _resolve_tier_bits(self) -> int:
        tier_cutoff_1 = self.total_layers // 3
        tier_cutoff_2 = (2 * self.total_layers) // 3
        if self.layer_idx < tier_cutoff_1:
            return self.shallow_bits
        if self.layer_idx < tier_cutoff_2:
            return self.mid_bits
        return self.deep_bits

    def _packing_spec(self, head_dim: int) -> tuple[int, int, Any]:
        """Return (pack_bits, packed_cols, storage_dtype) for this layer tier."""
        bits = self.effective_bits
        if bits == 4 and head_dim % 2 == 0:
            return 4, head_dim // 2, torch.uint8
        if bits == 2 and head_dim % 4 == 0:
            return 2, head_dim // 4, torch.uint8
        return 8, head_dim, torch.int8

    def _pack_quantized(self, deltas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize delta tensor to layer tier bitwidth and bit-pack along head_dim."""
        bits = self.effective_bits
        max_val = (1 << (bits - 1)) - 1
        min_val = -(1 << (bits - 1))
        scale = deltas.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / float(max_val)
        q = torch.round(deltas / scale).clamp(min_val, max_val)

        if self.pack_bits == 4:
            u = (q.to(torch.int16) + 8).to(torch.uint8).reshape(*deltas.shape[:-1], -1, 2)
            packed = u[..., 0] | (u[..., 1] << 4)
            return packed, scale.to(deltas.dtype)
        if self.pack_bits == 2:
            u = (q.to(torch.int16) + 2).to(torch.uint8).reshape(*deltas.shape[:-1], -1, 4)
            packed = u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)
            return packed, scale.to(deltas.dtype)
        return q.to(torch.int8), scale.to(deltas.dtype)

    def _unpack_quantized(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack stored delta integers directly into self.target_dtype."""
        if self.pack_bits == 4:
            lo = (packed & 0x0F).to(torch.int16) - 8
            hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
            return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], self.head_dim).to(self.target_dtype)
        if self.pack_bits == 2:
            q0 = (packed & 0x03).to(torch.int16) - 2
            q1 = ((packed >> 2) & 0x03).to(torch.int16) - 2
            q2 = ((packed >> 4) & 0x03).to(torch.int16) - 2
            q3 = ((packed >> 6) & 0x03).to(torch.int16) - 2
            return torch.stack((q0, q1, q2, q3), dim=-1).reshape(*packed.shape[:-1], self.head_dim).to(self.target_dtype)
        return packed.to(self.target_dtype)

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

        recon = self._unpack_quantized(dq)
        recon.mul_(ds)

        b, h, _, d = anchors.shape
        full_groups = t // g
        full_tokens = full_groups * g
        if full_groups > 0:
            recon[:, :, :full_tokens, :].view(b, h, full_groups, g, d).add_(
                anchors[:, :, :full_groups, :].unsqueeze(-2)
            )
        if full_tokens < t:
            recon[:, :, full_tokens:, :].add_(anchors[:, :, full_groups : full_groups + 1, :])
        return recon

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
        self.pack_bits, _, _ = self._packing_spec(self.head_dim)
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

        if required_capacity <= 2048:
            new_capacity = max(required_capacity, self.capacity * 2, 1024)
        else:
            # Linear headroom past 2K avoids 16K->32K->64K doubling spikes under 16GB cap
            new_capacity = max(required_capacity + 256, self.capacity)

        new_group_cap = math.ceil(new_capacity / self.anchor_group_size)
        _, q_cols, q_dtype = self._packing_spec(head_dim)

        saved_cpu = None
        if self.total_tokens_seen > 0 and self.k_anchors_buf is not None:
            t = self.total_tokens_seen
            g_used = math.ceil(t / self.anchor_group_size)
            saved_cpu = (
                t,
                g_used,
                self.k_anchors_buf[:, :, :g_used].cpu(),
                self.k_delta_q_buf[:, :, :t].cpu(),
                self.k_delta_s_buf[:, :, :t].cpu(),
                self.v_anchors_buf[:, :, :g_used].cpu(),
                self.v_delta_q_buf[:, :, :t].cpu(),
                self.v_delta_s_buf[:, :, :t].cpu(),
            )

        # Drop old GPU buffers before allocating larger ones to avoid 2x VRAM peak
        self.k_anchors_buf = None
        self.k_delta_q_buf = None
        self.k_delta_s_buf = None
        self.v_anchors_buf = None
        self.v_delta_q_buf = None
        self.v_delta_s_buf = None

        self.k_anchors_buf = torch.empty((batch_size, num_kv, new_group_cap, head_dim), dtype=target_dtype, device=target_device)
        self.k_delta_q_buf = torch.empty((batch_size, num_kv, new_capacity, q_cols), dtype=q_dtype, device=target_device)
        self.k_delta_s_buf = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)

        self.v_anchors_buf = torch.empty_like(self.k_anchors_buf)
        self.v_delta_q_buf = torch.empty_like(self.k_delta_q_buf)
        self.v_delta_s_buf = torch.empty_like(self.k_delta_s_buf)

        if saved_cpu is not None:
            t, g_used, ka, kdq, kds, va, vdq, vds = saved_cpu
            self.k_anchors_buf[:, :, :g_used].copy_(ka, non_blocking=False)
            self.k_delta_q_buf[:, :, :t].copy_(kdq, non_blocking=False)
            self.k_delta_s_buf[:, :, :t].copy_(kds, non_blocking=False)
            self.v_anchors_buf[:, :, :g_used].copy_(va, non_blocking=False)
            self.v_delta_q_buf[:, :, :t].copy_(vdq, non_blocking=False)
            self.v_delta_s_buf[:, :, :t].copy_(vds, non_blocking=False)

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
        self.effective_bits = self._resolve_tier_bits()
        self.pack_bits, _, _ = self._packing_spec(head_dim)

        curr = self.total_tokens_seen
        self._ensure_capacity(
            batch_size, num_kv, head_dim, key_states.device, key_states.dtype, curr + new_tokens
        )

        g = self.anchor_group_size
        positions = torch.arange(curr, curr + new_tokens, device=key_states.device)
        is_anchor = (positions % g) == 0
        if is_anchor.any():
            anchor_local_idx = torch.nonzero(is_anchor, as_tuple=False).squeeze(-1)
            anchor_group_idx = positions[anchor_local_idx] // g
            self.k_anchors_buf[:, :, anchor_group_idx, :] = key_states[:, :, anchor_local_idx, :]
            self.v_anchors_buf[:, :, anchor_group_idx, :] = value_states[:, :, anchor_local_idx, :]

        group_ids = positions // g

        # Process K then V sequentially to minimize peak temporary tensor footprint during prefill
        active_k_anchors = self.k_anchors_buf[:, :, group_ids, :]
        k_deltas = key_states - active_k_anchors
        del active_k_anchors
        k_q, k_s = self._pack_quantized(k_deltas)
        del k_deltas
        self.k_delta_q_buf[:, :, curr : curr + new_tokens].copy_(k_q)
        self.k_delta_s_buf[:, :, curr : curr + new_tokens].copy_(k_s)
        del k_q, k_s

        active_v_anchors = self.v_anchors_buf[:, :, group_ids, :]
        v_deltas = value_states - active_v_anchors
        del active_v_anchors
        v_q, v_s = self._pack_quantized(v_deltas)
        del v_deltas
        self.v_delta_q_buf[:, :, curr : curr + new_tokens].copy_(v_q)
        self.v_delta_s_buf[:, :, curr : curr + new_tokens].copy_(v_s)
        del v_q, v_s

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
        self.capacity = 0
        self.group_capacity = 0
        self.k_anchors_buf = None
        self.k_delta_q_buf = None
        self.k_delta_s_buf = None
        self.v_anchors_buf = None
        self.v_delta_q_buf = None
        self.v_delta_s_buf = None
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
