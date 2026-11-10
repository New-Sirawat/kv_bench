from __future__ import annotations

import weakref
from collections.abc import Iterator
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer
from src.techniques.turboquant import (
    dequantize_key_4bit,
    dequantize_value_3plus1bit,
    quantize_key_4bit,
    quantize_value_3plus1bit,
    turboquant_rotate,
    turboquant_unrotate,
)

_ACTIVE_TURBOQUANT_CACHE_REF: weakref.ReferenceType[TurboQuantCache] | None = None


def get_active_turboquant_cache() -> TurboQuantCache | None:
    """Return currently active TurboQuantCache instance if alive, else None."""
    global _ACTIVE_TURBOQUANT_CACHE_REF
    if _ACTIVE_TURBOQUANT_CACHE_REF is not None:
        return _ACTIVE_TURBOQUANT_CACHE_REF()
    return None


def set_active_turboquant_cache(cache: TurboQuantCache | None) -> None:
    """Register weak reference to active TurboQuantCache ensuring zero tensor retention leaks."""
    global _ACTIVE_TURBOQUANT_CACHE_REF
    if cache is not None:
        _ACTIVE_TURBOQUANT_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_TURBOQUANT_CACHE_REF = None


class TurboQuantLayer(DynamicLayer):
    """Cache layer implementing TurboQuant asymmetric FWHT 4-bit Key and 3+1-bit QJL Value storage."""

    is_sliding = False

    def __init__(self, rotate_seed: int = 42, sketch_seed: int = 1337):
        self._keys_fallback: torch.Tensor | None = None
        self._values_fallback: torch.Tensor | None = None

        self.rotate_seed = int(rotate_seed)
        self.sketch_seed = int(sketch_seed)
        self.capacity: int = 0
        self.total_tokens_seen: int = 0
        self.head_dim: int = 0
        self.target_dtype: Any = None
        self.target_device: Any = None

        # Pre-allocated rotated & packed KV buffers
        self.k_packed_buf: torch.Tensor | None = None
        self.k_scales_buf: torch.Tensor | None = None
        self.k_norms_buf: torch.Tensor | None = None

        self.v_scalar3_buf: torch.Tensor | None = None
        self.v_sketch_buf: torch.Tensor | None = None
        self.v_scale3_buf: torch.Tensor | None = None
        self.v_rnorm_buf: torch.Tensor | None = None
        self.v_norms_buf: torch.Tensor | None = None

        super().__init__()

    @property
    def original_positions(self) -> list[int]:
        """All ingested token positions are physically preserved in compressed storage."""
        return list(range(self.total_tokens_seen))

    @property
    def keys(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._keys_fallback
        assert self.k_packed_buf is not None and self.k_scales_buf is not None and self.k_norms_buf is not None
        kp = self.k_packed_buf[:, :, : self.total_tokens_seen, :]
        ks = self.k_scales_buf[:, :, : self.total_tokens_seen, :]
        kn = self.k_norms_buf[:, :, : self.total_tokens_seen, :]
        k_rot_deq = dequantize_key_4bit(kp, ks, original_dim=self.head_dim, dtype=self.target_dtype)
        return turboquant_unrotate(k_rot_deq, kn, seed=self.rotate_seed).to(self.target_dtype)

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        self._keys_fallback = value

    @property
    def values(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._values_fallback
        assert (
            self.v_scalar3_buf is not None
            and self.v_sketch_buf is not None
            and self.v_scale3_buf is not None
            and self.v_rnorm_buf is not None
            and self.v_norms_buf is not None
        )
        vs3 = self.v_scalar3_buf[:, :, : self.total_tokens_seen, :]
        vsk = self.v_sketch_buf[:, :, : self.total_tokens_seen, :]
        vsc = self.v_scale3_buf[:, :, : self.total_tokens_seen, :]
        vrn = self.v_rnorm_buf[:, :, : self.total_tokens_seen, :]
        vn = self.v_norms_buf[:, :, : self.total_tokens_seen, :]
        v_rot_deq = dequantize_value_3plus1bit(
            vs3,
            vsk,
            vsc,
            vrn,
            original_dim=self.head_dim,
            sketch_seed=self.sketch_seed,
            dtype=self.target_dtype,
        )
        return turboquant_unrotate(v_rot_deq, vn, seed=self.rotate_seed).to(self.target_dtype)

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

        new_kp = torch.empty((batch_size, num_kv, new_capacity, head_dim // 2), dtype=torch.uint8, device=target_device)
        new_ks = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)
        new_kn = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)

        new_vs3 = torch.empty((batch_size, num_kv, new_capacity, head_dim // 2), dtype=torch.uint8, device=target_device)
        new_vsk = torch.empty((batch_size, num_kv, new_capacity, head_dim // 8), dtype=torch.uint8, device=target_device)
        new_vsc = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)
        new_vrn = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)
        new_vn = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)

        if self.total_tokens_seen > 0 and self.k_packed_buf is not None:
            t = self.total_tokens_seen
            new_kp[:, :, :t].copy_(self.k_packed_buf[:, :, :t])
            new_ks[:, :, :t].copy_(self.k_scales_buf[:, :, :t])
            new_kn[:, :, :t].copy_(self.k_norms_buf[:, :, :t])

            new_vs3[:, :, :t].copy_(self.v_scalar3_buf[:, :, :t])
            new_vsk[:, :, :t].copy_(self.v_sketch_buf[:, :, :t])
            new_vsc[:, :, :t].copy_(self.v_scale3_buf[:, :, :t])
            new_vrn[:, :, :t].copy_(self.v_rnorm_buf[:, :, :t])
            new_vn[:, :, :t].copy_(self.v_norms_buf[:, :, :t])

        self.k_packed_buf = new_kp
        self.k_scales_buf = new_ks
        self.k_norms_buf = new_kn
        self.v_scalar3_buf = new_vs3
        self.v_sketch_buf = new_vsk
        self.v_scale3_buf = new_vsc
        self.v_rnorm_buf = new_vrn
        self.v_norms_buf = new_vn
        self.capacity = new_capacity

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

        k_rot, k_norms = turboquant_rotate(key_states, seed=self.rotate_seed)
        k_packed, k_scales, _ = quantize_key_4bit(k_rot)

        v_rot, v_norms = turboquant_rotate(value_states, seed=self.rotate_seed)
        v_s3, v_sk, v_sc, v_rn = quantize_value_3plus1bit(v_rot, sketch_seed=self.sketch_seed)

        curr = self.total_tokens_seen
        self._ensure_capacity(
            batch_size, num_kv, head_dim, key_states.device, key_states.dtype, curr + new_tokens
        )

        self.k_packed_buf[:, :, curr : curr + new_tokens].copy_(k_packed)
        self.k_scales_buf[:, :, curr : curr + new_tokens].copy_(k_scales.to(key_states.dtype))
        self.k_norms_buf[:, :, curr : curr + new_tokens].copy_(k_norms.to(key_states.dtype))

        self.v_scalar3_buf[:, :, curr : curr + new_tokens].copy_(v_s3)
        self.v_sketch_buf[:, :, curr : curr + new_tokens].copy_(v_sk)
        self.v_scale3_buf[:, :, curr : curr + new_tokens].copy_(v_sc.to(key_states.dtype))
        self.v_rnorm_buf[:, :, curr : curr + new_tokens].copy_(v_rn.to(key_states.dtype))
        self.v_norms_buf[:, :, curr : curr + new_tokens].copy_(v_norms.to(key_states.dtype))

        self.total_tokens_seen += new_tokens

        recon_k = self.keys
        recon_v = self.values
        assert recon_k is not None and recon_v is not None
        return recon_k, recon_v

    @property
    def quantized_bytes(self) -> int:
        if self.total_tokens_seen == 0 or self.k_packed_buf is None:
            return 0
        t = self.total_tokens_seen
        tensors = [
            self.k_packed_buf[:, :, :t],
            self.k_scales_buf[:, :, :t],
            self.k_norms_buf[:, :, :t],
            self.v_scalar3_buf[:, :, :t],
            self.v_sketch_buf[:, :, :t],
            self.v_scale3_buf[:, :, :t],
            self.v_rnorm_buf[:, :, :t],
            self.v_norms_buf[:, :, :t],
        ]
        return sum(x.nelement() * x.element_size() for x in tensors)

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


class TurboQuantCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing TurboQuant asymmetric KV compression."""

    is_turboquant = True

    def __init__(self, rotate_seed: int = 42, sketch_seed: int = 1337):
        self.rotate_seed = int(rotate_seed)
        self.sketch_seed = int(sketch_seed)
        super().__init__(
            layer_class_to_replicate=lambda: TurboQuantLayer(
                rotate_seed=self.rotate_seed,
                sketch_seed=self.sketch_seed,
            )
        )
        set_active_turboquant_cache(self)

    @property
    def original_positions(self) -> list[int]:
        if not self.layers:
            return []
        return self.layers[0].original_positions

    @property
    def total_quantized_bytes(self) -> int:
        return sum(getattr(layer, "quantized_bytes", 0) for layer in self.layers)

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
