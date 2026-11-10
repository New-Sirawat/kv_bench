from __future__ import annotations

import weakref
from collections.abc import Iterator
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer
from src.techniques.headq import (
    compute_key_sidecode,
    compute_query_pca_basis,
    dequantize_key_2bit,
    dequantize_key_sidecode,
    dequantize_value_4bit,
    quantize_key_2bit,
    quantize_value_4bit,
)

_ACTIVE_HEADQ_CACHE_REF: weakref.ReferenceType[HeadQCache] | None = None


def get_active_headq_cache() -> HeadQCache | None:
    """Return currently active HeadQCache instance if alive, else None."""
    global _ACTIVE_HEADQ_CACHE_REF
    if _ACTIVE_HEADQ_CACHE_REF is not None:
        return _ACTIVE_HEADQ_CACHE_REF()
    return None


def set_active_headq_cache(cache: HeadQCache | None) -> None:
    """Register weak reference to active HeadQCache ensuring zero tensor retention leaks."""
    global _ACTIVE_HEADQ_CACHE_REF
    if cache is not None:
        _ACTIVE_HEADQ_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_HEADQ_CACHE_REF = None


class HeadQLayer(DynamicLayer):
    """Cache layer implementing HeadQ 2-bit base Key + rank-r residual side-code + 4-bit Value storage."""

    is_sliding = False

    def __init__(self, rank: int = 4, sidecode_bits: int = 4):
        self._keys_fallback: torch.Tensor | None = None
        self._values_fallback: torch.Tensor | None = None

        self.rank = int(rank)
        self.sidecode_bits = int(sidecode_bits)
        self.capacity: int = 0
        self.total_tokens_seen: int = 0
        self.head_dim: int = 0
        self.target_dtype: Any = None
        self.target_device: Any = None

        self.basis: torch.Tensor | None = None
        self.k_packed_buf: torch.Tensor | None = None
        self.k_scales_buf: torch.Tensor | None = None
        self.z_packed_buf: torch.Tensor | None = None
        self.z_scales_buf: torch.Tensor | None = None
        self.v_packed_buf: torch.Tensor | None = None
        self.v_scales_buf: torch.Tensor | None = None

        super().__init__()

        if self.rank <= 0 or (self.rank % 2 != 0):
            raise ValueError(f"rank must be a positive even integer for sidecode packing, got {rank}")

    @property
    def original_positions(self) -> list[int]:
        return list(range(self.total_tokens_seen))

    @property
    def keys(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._keys_fallback
        assert (
            self.k_packed_buf is not None
            and self.k_scales_buf is not None
            and self.z_packed_buf is not None
            and self.z_scales_buf is not None
            and self.basis is not None
        )
        t = self.total_tokens_seen
        k_base = dequantize_key_2bit(
            self.k_packed_buf[:, :, :t, :],
            self.k_scales_buf[:, :, :t, :],
            original_dim=self.head_dim,
            dtype=self.target_dtype,
        )
        z_recon = dequantize_key_sidecode(
            self.z_packed_buf[:, :, :t, :],
            self.z_scales_buf[:, :, :t, :],
            rank=self.rank,
            dtype=self.target_dtype,
        )
        k_corr = torch.matmul(z_recon.float(), self.basis.float().transpose(0, 1)).to(self.target_dtype)
        return (k_base + k_corr).to(self.target_dtype)

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        self._keys_fallback = value

    @property
    def values(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._values_fallback
        assert self.v_packed_buf is not None and self.v_scales_buf is not None
        t = self.total_tokens_seen
        return dequantize_value_4bit(
            self.v_packed_buf[:, :, :t, :],
            self.v_scales_buf[:, :, :t, :],
            original_dim=self.head_dim,
            dtype=self.target_dtype,
        )

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

        new_kp = torch.empty((batch_size, num_kv, new_capacity, head_dim // 4), dtype=torch.uint8, device=target_device)
        new_ks = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)
        new_zp = torch.empty((batch_size, num_kv, new_capacity, self.rank // 2), dtype=torch.uint8, device=target_device)
        new_zs = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)
        new_vp = torch.empty((batch_size, num_kv, new_capacity, head_dim // 2), dtype=torch.uint8, device=target_device)
        new_vs = torch.empty((batch_size, num_kv, new_capacity, 1), dtype=target_dtype, device=target_device)

        if self.total_tokens_seen > 0 and self.k_packed_buf is not None:
            t = self.total_tokens_seen
            new_kp[:, :, :t].copy_(self.k_packed_buf[:, :, :t])
            new_ks[:, :, :t].copy_(self.k_scales_buf[:, :, :t])
            new_zp[:, :, :t].copy_(self.z_packed_buf[:, :, :t])
            new_zs[:, :, :t].copy_(self.z_scales_buf[:, :, :t])
            new_vp[:, :, :t].copy_(self.v_packed_buf[:, :, :t])
            new_vs[:, :, :t].copy_(self.v_scales_buf[:, :, :t])

        self.k_packed_buf = new_kp
        self.k_scales_buf = new_ks
        self.z_packed_buf = new_zp
        self.z_scales_buf = new_zs
        self.v_packed_buf = new_vp
        self.v_scales_buf = new_vs
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

        k_packed, k_scales, k_dequant = quantize_key_2bit(key_states)
        k_residual = key_states - k_dequant

        if self.basis is None:
            self.basis = compute_query_pca_basis(k_residual, rank=self.rank)

        z_packed, z_scales, _ = compute_key_sidecode(
            k_residual, self.basis, sidecode_bits=self.sidecode_bits
        )
        v_packed, v_scales, _ = quantize_value_4bit(value_states)

        curr = self.total_tokens_seen
        self._ensure_capacity(
            batch_size, num_kv, head_dim, key_states.device, key_states.dtype, curr + new_tokens
        )

        self.k_packed_buf[:, :, curr : curr + new_tokens].copy_(k_packed)
        self.k_scales_buf[:, :, curr : curr + new_tokens].copy_(k_scales.to(key_states.dtype))
        self.z_packed_buf[:, :, curr : curr + new_tokens].copy_(z_packed)
        self.z_scales_buf[:, :, curr : curr + new_tokens].copy_(z_scales.to(key_states.dtype))
        self.v_packed_buf[:, :, curr : curr + new_tokens].copy_(v_packed)
        self.v_scales_buf[:, :, curr : curr + new_tokens].copy_(v_scales.to(value_states.dtype))

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
            self.z_packed_buf[:, :, :t],
            self.z_scales_buf[:, :, :t],
            self.v_packed_buf[:, :, :t],
            self.v_scales_buf[:, :, :t],
        ]
        basis_bytes = self.basis.nelement() * self.basis.element_size() if self.basis is not None else 0
        return basis_bytes + sum(x.nelement() * x.element_size() for x in tensors)

    def get_seq_length(self) -> int:
        return self.total_tokens_seen

    def get_max_length(self) -> int:
        return max(self.total_tokens_seen, 32768)

    def reset(self) -> None:
        self.total_tokens_seen = 0
        self.basis = None
        self.is_initialized = False
        self._keys_fallback = None
        self._values_fallback = None
        if hasattr(self, "cumulative_length"):
            if isinstance(self.cumulative_length, int):
                self.cumulative_length = 0
            else:
                self.cumulative_length.zero_()


class HeadQCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing HeadQ 2-bit Key + PCA side-code storage."""

    is_headq = True

    def __init__(self, rank: int = 4, sidecode_bits: int = 4):
        self.rank = int(rank)
        self.sidecode_bits = int(sidecode_bits)
        super().__init__(
            layer_class_to_replicate=lambda: HeadQLayer(
                rank=self.rank,
                sidecode_bits=self.sidecode_bits,
            )
        )
        set_active_headq_cache(self)

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
