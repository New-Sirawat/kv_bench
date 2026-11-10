from __future__ import annotations

import weakref
from collections.abc import Iterator
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer
from src.techniques.qserve_kv4 import dequantize_kv_int4, quantize_kv_int4

_ACTIVE_QSERVE_CACHE_REF: weakref.ReferenceType[QServeKVCache] | None = None


def get_active_qserve_cache() -> QServeKVCache | None:
    """Return currently active QServeKVCache instance if alive, else None."""
    global _ACTIVE_QSERVE_CACHE_REF
    if _ACTIVE_QSERVE_CACHE_REF is not None:
        return _ACTIVE_QSERVE_CACHE_REF()
    return None


def set_active_qserve_cache(cache: QServeKVCache | None) -> None:
    """Register weak reference to active QServeKVCache ensuring zero tensor retention leaks."""
    global _ACTIVE_QSERVE_CACHE_REF
    if cache is not None:
        _ACTIVE_QSERVE_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_QSERVE_CACHE_REF = None


class QServeKVLayer(DynamicLayer):
    """Cache layer implementing QServe SmoothAttention + symmetric per-group INT4 KV storage.

    Persistent state resides strictly in packed uint8 (4-bit) buffers and compact FP16/BF16
    group scales using pre-allocated capacity buffers to eliminate decode-loop fragmentation.
    Full floating-point tensors are reconstructed cleanly on demand for Hugging Face SDPA.
    """

    is_sliding = False

    def __init__(self, group_size: int = 32, alpha: float = 0.5):
        self._keys_fallback: torch.Tensor | None = None
        self._values_fallback: torch.Tensor | None = None

        self.group_size = int(group_size)
        self.alpha = float(alpha)

        self.capacity: int = 0
        self.total_tokens_seen: int = 0
        self.head_dim: int = 0
        self.target_dtype: Any = None
        self.target_device: Any = None

        # Pre-allocated packed 4-bit buffers on active device
        self.smooth_k: torch.Tensor | None = None
        self.k_packed_buf: torch.Tensor | None = None
        self.k_scales_buf: torch.Tensor | None = None
        self.v_packed_buf: torch.Tensor | None = None
        self.v_scales_buf: torch.Tensor | None = None

        super().__init__()

        if self.group_size <= 0 or (self.group_size % 2 != 0):
            raise ValueError(f"group_size must be a positive even integer, got {group_size}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

    @property
    def original_positions(self) -> list[int]:
        """All ingested token positions are physically preserved in quantized INT4 storage."""
        return list(range(self.total_tokens_seen))

    @property
    def keys(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._keys_fallback
        assert self.k_packed_buf is not None and self.k_scales_buf is not None
        p_view = self.k_packed_buf[:, :, : self.total_tokens_seen, :]
        s_view = self.k_scales_buf[:, :, : self.total_tokens_seen, :, :]
        recon = dequantize_kv_int4(
            p_view, s_view, original_dtype=self.target_dtype, group_size=self.group_size
        )
        if self.smooth_k is not None:
            recon = (recon * self.smooth_k).to(self.target_dtype)
        return recon

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        self._keys_fallback = value

    @property
    def values(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._values_fallback
        assert self.v_packed_buf is not None and self.v_scales_buf is not None
        p_view = self.v_packed_buf[:, :, : self.total_tokens_seen, :]
        s_view = self.v_scales_buf[:, :, : self.total_tokens_seen, :, :]
        return dequantize_kv_int4(
            p_view, s_view, original_dtype=self.target_dtype, group_size=self.group_size
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
        packed_dim = head_dim // 2
        num_groups = head_dim // self.group_size

        new_k_p = torch.empty(
            (batch_size, num_kv, new_capacity, packed_dim),
            dtype=torch.uint8,
            device=target_device,
        )
        new_k_s = torch.empty(
            (batch_size, num_kv, new_capacity, num_groups, 1),
            dtype=target_dtype,
            device=target_device,
        )
        new_v_p = torch.empty_like(new_k_p)
        new_v_s = torch.empty_like(new_k_s)

        if self.total_tokens_seen > 0 and self.k_packed_buf is not None:
            new_k_p[:, :, : self.total_tokens_seen].copy_(self.k_packed_buf[:, :, : self.total_tokens_seen])
            new_k_s[:, :, : self.total_tokens_seen].copy_(self.k_scales_buf[:, :, : self.total_tokens_seen])
            new_v_p[:, :, : self.total_tokens_seen].copy_(self.v_packed_buf[:, :, : self.total_tokens_seen])
            new_v_s[:, :, : self.total_tokens_seen].copy_(self.v_scales_buf[:, :, : self.total_tokens_seen])

        self.k_packed_buf = new_k_p
        self.k_scales_buf = new_k_s
        self.v_packed_buf = new_v_p
        self.v_scales_buf = new_v_s
        self.capacity = new_capacity

    def _ensure_smooth_k(self, key_states: torch.Tensor) -> None:
        if self.smooth_k is not None:
            return

        if self.alpha == 0.0:
            self.smooth_k = torch.ones(
                (1, key_states.shape[1], 1, key_states.shape[-1]),
                dtype=key_states.dtype,
                device=key_states.device,
            )
            return

        batch, num_kv, _, head_dim = key_states.shape
        num_groups = head_dim // self.group_size
        # Compute per-channel peak magnitude relative to each quantization group's mean
        k_abs_max = key_states.float().abs().amax(dim=(0, 2), keepdim=True).clamp(min=1e-5)
        k_grouped = k_abs_max.reshape(1, num_kv, 1, num_groups, self.group_size)
        group_mean = k_grouped.mean(dim=-1, keepdim=True).clamp(min=1e-5)
        smooth_ratio = (k_grouped / group_mean).clamp(min=1e-3, max=1e3)
        self.smooth_k = smooth_ratio.pow(self.alpha).reshape(1, num_kv, 1, head_dim).to(key_states.dtype)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize incoming KV states with SmoothAttention + INT4 and reconstruct working tensors."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        batch_size = key_states.shape[0]
        num_kv = key_states.shape[1]
        new_tokens = key_states.shape[-2]
        head_dim = key_states.shape[-1]

        self.target_dtype = key_states.dtype
        self.target_device = key_states.device
        self.head_dim = head_dim

        self._ensure_smooth_k(key_states)
        assert self.smooth_k is not None
        k_smooth = (key_states / self.smooth_k).to(key_states.dtype)

        k_p, k_s = quantize_kv_int4(k_smooth, group_size=self.group_size)
        v_p, v_s = quantize_kv_int4(value_states, group_size=self.group_size)

        curr = self.total_tokens_seen
        self._ensure_capacity(
            batch_size, num_kv, head_dim, key_states.device, key_states.dtype, curr + new_tokens
        )

        assert self.k_packed_buf is not None and self.k_scales_buf is not None
        assert self.v_packed_buf is not None and self.v_scales_buf is not None

        self.k_packed_buf[:, :, curr : curr + new_tokens].copy_(k_p)
        self.k_scales_buf[:, :, curr : curr + new_tokens].copy_(k_s)
        self.v_packed_buf[:, :, curr : curr + new_tokens].copy_(v_p)
        self.v_scales_buf[:, :, curr : curr + new_tokens].copy_(v_s)

        self.total_tokens_seen += new_tokens

        recon_k = self.keys
        recon_v = self.values
        assert recon_k is not None and recon_v is not None
        return recon_k, recon_v

    @property
    def quantized_bytes(self) -> int:
        if self.total_tokens_seen == 0 or self.k_packed_buf is None:
            return 0
        kp = self.k_packed_buf[:, :, : self.total_tokens_seen]
        ks = self.k_scales_buf[:, :, : self.total_tokens_seen]
        vp = self.v_packed_buf[:, :, : self.total_tokens_seen]
        vs = self.v_scales_buf[:, :, : self.total_tokens_seen]
        return (
            kp.nelement() * kp.element_size()
            + ks.nelement() * ks.element_size()
            + vp.nelement() * vp.element_size()
            + vs.nelement() * vs.element_size()
        )

    def get_seq_length(self) -> int:
        return self.total_tokens_seen

    def get_max_length(self) -> int:
        return max(self.total_tokens_seen, 32768)

    def reset(self) -> None:
        self.total_tokens_seen = 0
        self.smooth_k = None
        self.is_initialized = False
        self._keys_fallback = None
        self._values_fallback = None
        if hasattr(self, "cumulative_length"):
            if isinstance(self.cumulative_length, int):
                self.cumulative_length = 0
            else:
                self.cumulative_length.zero_()


class QServeKVCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing QServe SmoothAttention + INT4 KV storage."""

    is_qserve = True

    def __init__(self, group_size: int = 32, alpha: float = 0.5):
        if int(group_size) <= 0 or (int(group_size) % 2 != 0):
            raise ValueError(f"group_size must be a positive even integer, got {group_size}")
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

        self.group_size = int(group_size)
        self.alpha = float(alpha)

        super().__init__(
            layer_class_to_replicate=lambda: QServeKVLayer(
                group_size=self.group_size,
                alpha=self.alpha,
            )
        )
        set_active_qserve_cache(self)

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
