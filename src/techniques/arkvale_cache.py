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

_ACTIVE_ARKVALE_CACHE_REF: weakref.ReferenceType[ArkValeCache] | None = None


def get_active_arkvale_cache() -> ArkValeCache | None:
    """Return currently active ArkValeCache instance if alive, else None."""
    global _ACTIVE_ARKVALE_CACHE_REF
    if _ACTIVE_ARKVALE_CACHE_REF is not None:
        return _ACTIVE_ARKVALE_CACHE_REF()
    return None


def set_active_arkvale_cache(cache: ArkValeCache | None) -> None:
    """Register weak reference to active ArkValeCache ensuring zero tensor retention leaks."""
    global _ACTIVE_ARKVALE_CACHE_REF
    if cache is not None:
        _ACTIVE_ARKVALE_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_ARKVALE_CACHE_REF = None


class ArkValeLayer(DynamicLayer):
    """Cache layer implementing ArkVale page-level backup, cached GPU AABB digests, and page recall."""

    is_sliding = False

    def __init__(
        self,
        layer_idx: int = 0,
        page_size: int = 16,
        active_pages_budget: int = 16,
        exempt_early_layers: int = 2,
    ):
        self._keys_fallback: torch.Tensor | None = None
        self._values_fallback: torch.Tensor | None = None

        self.layer_idx = int(layer_idx)
        self.page_size = int(page_size)
        self.active_pages_budget = int(active_pages_budget)
        self.exempt_early_layers = int(exempt_early_layers)

        self.capacity: int = 0
        self.page_capacity: int = 0
        self.total_tokens_seen: int = 0
        self.head_dim: int = 0
        self.target_dtype: Any = None
        self.target_device: Any = None

        # Host pinned CPU full backup pool
        self.cpu_k_buf: torch.Tensor | None = None
        self.cpu_v_buf: torch.Tensor | None = None

        # Pre-allocated GPU AABB page digests: [B, H_kv, max_pages, D]
        self.gpu_b_min: torch.Tensor | None = None
        self.gpu_b_max: torch.Tensor | None = None

        # Compact active sparse GPU subset during pruned decode
        self.gpu_active_k: torch.Tensor | None = None
        self.gpu_active_v: torch.Tensor | None = None

        self.active_positions: list[int] = []
        self.resident_pages: set[int] = set()
        self.total_bytes_transferred_pcie: int = 0
        self.total_recalled_pages: int = 0

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
        if self.gpu_active_k is not None:
            return self.gpu_active_k
        assert self.cpu_k_buf is not None
        return self.cpu_k_buf[:, :, : self.total_tokens_seen].to(self.target_device, non_blocking=True)

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        self._keys_fallback = value

    @property
    def values(self) -> torch.Tensor | None:
        if not self.is_initialized or self.total_tokens_seen == 0:
            return self._values_fallback
        if self.gpu_active_v is not None:
            return self.gpu_active_v
        assert self.cpu_v_buf is not None
        return self.cpu_v_buf[:, :, : self.total_tokens_seen].to(self.target_device, non_blocking=True)

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
        new_page_cap = math.ceil(new_capacity / self.page_size)
        pin = torch.cuda.is_available()

        new_cpu_k = torch.empty(
            (batch_size, num_kv, new_capacity, head_dim), dtype=target_dtype, device="cpu", pin_memory=pin
        )
        new_cpu_v = torch.empty_like(new_cpu_k)
        new_b_min = torch.zeros(
            (batch_size, num_kv, new_page_cap, head_dim), dtype=target_dtype, device=target_device
        )
        new_b_max = torch.zeros_like(new_b_min)

        if self.total_tokens_seen > 0 and self.cpu_k_buf is not None and self.cpu_v_buf is not None:
            t = self.total_tokens_seen
            p_used = math.ceil(t / self.page_size)
            new_cpu_k[:, :, :t].copy_(self.cpu_k_buf[:, :, :t])
            new_cpu_v[:, :, :t].copy_(self.cpu_v_buf[:, :, :t])
            if self.gpu_b_min is not None and self.gpu_b_max is not None:
                new_b_min[:, :, :p_used].copy_(self.gpu_b_min[:, :, :p_used])
                new_b_max[:, :, :p_used].copy_(self.gpu_b_max[:, :, :p_used])

        self.cpu_k_buf = new_cpu_k
        self.cpu_v_buf = new_cpu_v
        self.gpu_b_min = new_b_min
        self.gpu_b_max = new_b_max
        self.capacity = new_capacity
        self.page_capacity = new_page_cap

    def _update_incremental_digests(
        self, key_states: torch.Tensor, start_pos: int, end_pos: int
    ) -> None:
        """Incrementally compute and cache GPU AABB page digests only for touched pages."""
        assert self.cpu_k_buf is not None and self.gpu_b_min is not None and self.gpu_b_max is not None
        p_start = start_pos // self.page_size
        p_end = (end_pos - 1) // self.page_size

        for p in range(p_start, p_end + 1):
            t_s = p * self.page_size
            t_e = min(end_pos, (p + 1) * self.page_size)
            if t_s >= start_pos:
                page_k = key_states[:, :, t_s - start_pos : t_e - start_pos, :]
            else:
                # Partial page spans across boundary: pull only the single small page from CPU
                page_k = self.cpu_k_buf[:, :, t_s:t_e, :].to(self.target_device)

            c = 0.5 * (page_k.amin(dim=-2) + page_k.amax(dim=-2))
            r_mean = (page_k - c.unsqueeze(-2)).abs().mean(dim=-2)
            self.gpu_b_max[:, :, p] = c + r_mean
            self.gpu_b_min[:, :, p] = c - r_mean

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update ArkVale CPU backup and perform cached AABB digest page selection/recall on decode."""
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

        assert self.cpu_k_buf is not None and self.cpu_v_buf is not None
        self.cpu_k_buf[:, :, curr : curr + new_tokens].copy_(key_states, non_blocking=True)
        self.cpu_v_buf[:, :, curr : curr + new_tokens].copy_(value_states, non_blocking=True)
        self.total_bytes_transferred_pcie += (
            (key_states.nelement() + value_states.nelement()) * key_states.element_size()
        )

        new_total = curr + new_tokens
        self.total_tokens_seen = new_total
        self._update_incremental_digests(key_states, curr, new_total)

        is_exempt = self.layer_idx < self.exempt_early_layers
        total_pages = math.ceil(new_total / self.page_size)

        if not is_exempt and new_tokens == 1 and total_pages > self.active_pages_budget:
            assert self.gpu_b_min is not None and self.gpu_b_max is not None
            b_min = self.gpu_b_min[:, :, :total_pages, :]
            b_max = self.gpu_b_max[:, :, :total_pages, :]

            q_proxy = key_states.unsqueeze(-2)
            prod_min = q_proxy * b_min.unsqueeze(-3)
            prod_max = q_proxy * b_max.unsqueeze(-3)
            page_scores = torch.maximum(prod_min, prod_max).sum(dim=-1).mean(dim=(0, 1, 2))

            k_budget = max(1, min(self.active_pages_budget, total_pages) - 2)
            _, top_pages = torch.topk(page_scores, k=k_budget)
            selected_pages = set(top_pages.tolist()) | {0, total_pages - 1}

            recalled = selected_pages.difference(self.resident_pages) if self.resident_pages else set()
            self.resident_pages = selected_pages
            if recalled:
                self.total_recalled_pages += len(recalled)
                recalled_tokens = len(recalled) * self.page_size
                self.total_bytes_transferred_pcie += (
                    2 * batch_size * num_kv * recalled_tokens * head_dim * key_states.element_size()
                )

            fetch_indices: list[int] = []
            for p in sorted(selected_pages):
                s = p * self.page_size
                e = min(new_total, (p + 1) * self.page_size)
                fetch_indices.extend(range(s, e))

            self.active_positions = fetch_indices
            idx_cpu = torch.tensor(fetch_indices, dtype=torch.long)
            self.gpu_active_k = self.cpu_k_buf[:, :, idx_cpu].to(key_states.device, non_blocking=True)
            self.gpu_active_v = self.cpu_v_buf[:, :, idx_cpu].to(key_states.device, non_blocking=True)
            return self.gpu_active_k, self.gpu_active_v

        # Prefill or exempt layers: return reconstructed view without pinning redundant GPU copies on self
        self.resident_pages = set(range(total_pages))
        self.active_positions = list(range(new_total))
        self.gpu_active_k = None
        self.gpu_active_v = None
        k_out = self.cpu_k_buf[:, :, :new_total].to(key_states.device, non_blocking=True)
        v_out = self.cpu_v_buf[:, :, :new_total].to(key_states.device, non_blocking=True)
        return k_out, v_out

    def get_seq_length(self) -> int:
        return self.total_tokens_seen

    def get_max_length(self) -> int:
        return max(self.total_tokens_seen, 32768)

    def reset(self) -> None:
        self.total_tokens_seen = 0
        self.active_positions = []
        self.resident_pages = set()
        self.gpu_active_k = None
        self.gpu_active_v = None
        self.total_bytes_transferred_pcie = 0
        self.total_recalled_pages = 0
        self.is_initialized = False
        self._keys_fallback = None
        self._values_fallback = None
        if hasattr(self, "cumulative_length"):
            if isinstance(self.cumulative_length, int):
                self.cumulative_length = 0
            else:
                self.cumulative_length.zero_()


class ArkValeCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing ArkVale page recall."""

    is_arkvale = True

    def __init__(
        self,
        page_size: int = 16,
        active_pages_budget: int = 16,
        exempt_early_layers: int = 2,
    ):
        self.page_size = int(page_size)
        self.active_pages_budget = int(active_pages_budget)
        self.exempt_early_layers = int(exempt_early_layers)

        super().__init__(
            layer_class_to_replicate=lambda: ArkValeLayer(
                layer_idx=len(self.layers),
                page_size=self.page_size,
                active_pages_budget=self.active_pages_budget,
                exempt_early_layers=self.exempt_early_layers,
            )
        )
        set_active_arkvale_cache(self)

    @property
    def original_positions(self) -> list[int]:
        if not self.layers:
            return []
        return self.layers[-1].original_positions

    @property
    def total_bytes_transferred_pcie(self) -> int:
        return sum(getattr(layer, "total_bytes_transferred_pcie", 0) for layer in self.layers)

    @property
    def total_recalled_pages(self) -> int:
        return sum(getattr(layer, "total_recalled_pages", 0) for layer in self.layers)

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
