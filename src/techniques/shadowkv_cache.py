from __future__ import annotations

import math
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
from src.techniques.shadowkv import compute_pre_rope_svd

_ACTIVE_SHADOWKV_CACHE_REF: weakref.ReferenceType[ShadowKVCache] | None = None


def get_active_shadowkv_cache() -> ShadowKVCache | None:
    """Return currently active ShadowKVCache instance if alive, else None."""
    global _ACTIVE_SHADOWKV_CACHE_REF
    if _ACTIVE_SHADOWKV_CACHE_REF is not None:
        return _ACTIVE_SHADOWKV_CACHE_REF()
    return None


def set_active_shadowkv_cache(cache: ShadowKVCache | None) -> None:
    """Register weak reference to active ShadowKVCache ensuring zero tensor retention leaks."""
    global _ACTIVE_SHADOWKV_CACHE_REF
    if cache is not None:
        _ACTIVE_SHADOWKV_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_SHADOWKV_CACHE_REF = None


class ShadowKVLayer(DynamicLayer):
    """Cache layer implementing ShadowKV low-rank GPU Keys, chunk landmarks, and CPU Value offload."""

    is_sliding = False

    def __init__(
        self,
        rank: int = 16,
        chunk_size: int = 8,
        outlier_ratio: float = 0.05,
        sparse_budget_chunks: int = 16,
    ):
        self._keys_fallback: torch.Tensor | None = None
        self._values_fallback: torch.Tensor | None = None

        self.rank = int(rank)
        self.chunk_size = int(chunk_size)
        self.outlier_ratio = float(outlier_ratio)
        self.sparse_budget_chunks = int(sparse_budget_chunks)

        self.capacity: int = 0
        self.total_tokens_seen: int = 0
        self.head_dim: int = 0
        self.effective_rank: int = 0
        self.target_dtype: Any = None
        self.target_device: Any = None

        # GPU-resident compressed Key subspace: K approx A @ B
        self.gpu_k_A: torch.Tensor | None = None
        self.gpu_k_B: torch.Tensor | None = None

        # CPU pinned host DRAM Value pool
        self.cpu_v_buf: torch.Tensor | None = None

        # Active working set tracking for PR #43 diagnostic telemetry
        self.active_positions: list[int] = []
        self.last_selected_chunks: set[int] = set()
        self.total_bytes_transferred_pcie: int = 0

        super().__init__()

    @property
    def original_positions(self) -> list[int]:
        if not self.active_positions:
            return list(range(self.total_tokens_seen))
        return list(self.active_positions)

    @property
    def keys(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._keys_fallback
        assert self.gpu_k_A is not None and self.gpu_k_B is not None
        idx = self.active_positions or list(range(self.total_tokens_seen))
        idx_t = torch.tensor(idx, dtype=torch.long, device=self.target_device)
        a_sel = self.gpu_k_A[:, :, idx_t, :]
        return torch.matmul(a_sel.float(), self.gpu_k_B.float()).to(self.target_dtype)

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        self._keys_fallback = value

    @property
    def values(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._values_fallback
        assert self.cpu_v_buf is not None
        idx = self.active_positions or list(range(self.total_tokens_seen))
        idx_t = torch.tensor(idx, dtype=torch.long, device="cpu")
        return self.cpu_v_buf[:, :, idx_t, :].to(self.target_device, non_blocking=True)

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
        eff_rank: int,
        target_device: Any,
        target_dtype: Any,
        required_capacity: int,
    ) -> None:
        if self.capacity >= required_capacity:
            return

        new_capacity = max(required_capacity, self.capacity * 2, 1024)
        pin = torch.cuda.is_available()

        new_A = torch.empty((batch_size, num_kv, new_capacity, eff_rank), dtype=target_dtype, device=target_device)
        new_cpu_v = torch.empty(
            (batch_size, num_kv, new_capacity, head_dim), dtype=target_dtype, device="cpu", pin_memory=pin
        )

        if self.total_tokens_seen > 0 and self.gpu_k_A is not None and self.cpu_v_buf is not None:
            t = self.total_tokens_seen
            new_A[:, :, :t].copy_(self.gpu_k_A[:, :, :t])
            new_cpu_v[:, :, :t].copy_(self.cpu_v_buf[:, :, :t])

        self.gpu_k_A = new_A
        self.cpu_v_buf = new_cpu_v
        self.capacity = new_capacity

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update ShadowKV low-rank Key projections and pinned CPU Value pool."""
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

        # Initialize low-rank SVD subspace B on initial prefill
        if self.gpu_k_B is None:
            eff_rank = min(self.rank, head_dim)
            if new_tokens >= eff_rank:
                A_init, B_init = compute_pre_rope_svd(key_states, rank=eff_rank)
                self.effective_rank = eff_rank
                self.gpu_k_B = B_init
                self._ensure_capacity(
                    batch_size, num_kv, head_dim, eff_rank, key_states.device, key_states.dtype, curr + new_tokens
                )
                assert self.gpu_k_A is not None
                self.gpu_k_A[:, :, curr : curr + new_tokens].copy_(A_init)
            else:
                # Sequence too short for truncated SVD: fallback to identity basis slice
                eff_rank = min(self.rank, head_dim)
                self.effective_rank = eff_rank
                self.gpu_k_B = torch.eye(eff_rank, head_dim, dtype=key_states.dtype, device=key_states.device).expand(
                    batch_size, num_kv, eff_rank, head_dim
                ).contiguous()
                self._ensure_capacity(
                    batch_size, num_kv, head_dim, eff_rank, key_states.device, key_states.dtype, curr + new_tokens
                )
                assert self.gpu_k_A is not None
                proj = torch.matmul(key_states.float(), self.gpu_k_B.float().transpose(-2, -1)).to(key_states.dtype)
                self.gpu_k_A[:, :, curr : curr + new_tokens].copy_(proj)
        else:
            # ShadowKV+: project arriving tokens directly onto fixed prefill SVD basis B^T
            assert self.gpu_k_B is not None
            self._ensure_capacity(
                batch_size, num_kv, head_dim, self.effective_rank, key_states.device, key_states.dtype, curr + new_tokens
            )
            assert self.gpu_k_A is not None
            proj = torch.matmul(key_states.float(), self.gpu_k_B.float().transpose(-2, -1)).to(key_states.dtype)
            self.gpu_k_A[:, :, curr : curr + new_tokens].copy_(proj)

        # Offload Values over PCIe to pinned CPU host DRAM
        assert self.cpu_v_buf is not None
        self.cpu_v_buf[:, :, curr : curr + new_tokens].copy_(value_states, non_blocking=True)
        self.total_bytes_transferred_pcie += value_states.nelement() * value_states.element_size()

        self.total_tokens_seen += new_tokens
        threshold_tokens = self.sparse_budget_chunks * self.chunk_size

        # Sparse landmark selection + selective PCIe value prefetch during decode
        if new_tokens == 1 and self.total_tokens_seen > threshold_tokens:
            assert self.gpu_k_A is not None and self.gpu_k_B is not None
            k_full_recon = torch.matmul(
                self.gpu_k_A[:, :, : self.total_tokens_seen].float(), self.gpu_k_B.float()
            )
            num_chunks = math.ceil(self.total_tokens_seen / self.chunk_size)
            pad_len = num_chunks * self.chunk_size - self.total_tokens_seen
            k_pad = F.pad(k_full_recon, (0, 0, 0, pad_len)) if pad_len > 0 else k_full_recon
            landmarks = k_pad.reshape(batch_size, num_kv, num_chunks, self.chunk_size, head_dim).mean(dim=-2)

            # Proxy score using latest key normalized direction
            proxy_scores = torch.matmul(key_states.float(), landmarks.transpose(-2, -1)).mean(dim=(0, 1, 2))
            top_k = min(self.sparse_budget_chunks, num_chunks)
            _, top_chunks = torch.topk(proxy_scores, k=top_k)

            # Always anchor chunk 0 (sink) and recent chunk (current token)
            selected_set = set(top_chunks.tolist()) | {0, num_chunks - 1}
            missed_chunks = selected_set.difference(self.last_selected_chunks)
            self.last_selected_chunks = selected_set

            fetch_indices: list[int] = []
            for c in sorted(selected_set):
                s = c * self.chunk_size
                e = min(self.total_tokens_seen, (c + 1) * self.chunk_size)
                fetch_indices.extend(range(s, e))

            self.active_positions = fetch_indices
            idx_t = torch.tensor(fetch_indices, dtype=torch.long, device=key_states.device)
            k_active = k_full_recon[:, :, idx_t, :].to(key_states.dtype)

            # PCIe transfer only for chunks not cached temporally in GPU working set
            idx_cpu = torch.tensor(fetch_indices, dtype=torch.long)
            v_active = self.cpu_v_buf[:, :, idx_cpu, :].to(key_states.device, non_blocking=True)
            if missed_chunks:
                missed_tokens = len(missed_chunks) * self.chunk_size
                self.total_bytes_transferred_pcie += (
                    batch_size * num_kv * missed_tokens * head_dim * value_states.element_size()
                )

            return k_active, v_active

        # During prefill or short prompts, retain all positions
        self.active_positions = list(range(self.total_tokens_seen))
        assert self.gpu_k_A is not None and self.gpu_k_B is not None
        k_out = torch.matmul(
            self.gpu_k_A[:, :, : self.total_tokens_seen].float(), self.gpu_k_B.float()
        ).to(key_states.dtype)
        v_out = self.cpu_v_buf[:, :, : self.total_tokens_seen].to(key_states.device, non_blocking=True)
        return k_out, v_out

    def get_seq_length(self) -> int:
        return self.total_tokens_seen

    def get_max_length(self) -> int:
        return max(self.total_tokens_seen, 32768)

    def reset(self) -> None:
        self.total_tokens_seen = 0
        self.gpu_k_B = None
        self.active_positions = []
        self.last_selected_chunks = set()
        self.total_bytes_transferred_pcie = 0
        self.is_initialized = False
        self._keys_fallback = None
        self._values_fallback = None
        if hasattr(self, "cumulative_length"):
            if isinstance(self.cumulative_length, int):
                self.cumulative_length = 0
            else:
                self.cumulative_length.zero_()


class ShadowKVCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing ShadowKV GPU/CPU asymmetric tiering."""

    is_shadowkv = True

    def __init__(
        self,
        rank: int = 16,
        chunk_size: int = 8,
        outlier_ratio: float = 0.05,
        sparse_budget_chunks: int = 16,
    ):
        self.rank = int(rank)
        self.chunk_size = int(chunk_size)
        self.outlier_ratio = float(outlier_ratio)
        self.sparse_budget_chunks = int(sparse_budget_chunks)

        super().__init__(
            layer_class_to_replicate=lambda: ShadowKVLayer(
                rank=self.rank,
                chunk_size=self.chunk_size,
                outlier_ratio=self.outlier_ratio,
                sparse_budget_chunks=self.sparse_budget_chunks,
            )
        )
        set_active_shadowkv_cache(self)

    @property
    def original_positions(self) -> list[int]:
        if not self.layers:
            return []
        return self.layers[0].original_positions

    @property
    def total_bytes_transferred_pcie(self) -> int:
        return sum(getattr(layer, "total_bytes_transferred_pcie", 0) for layer in self.layers)

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
