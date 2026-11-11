from __future__ import annotations

import weakref
from collections.abc import Iterator
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer
from src.techniques.epicache import (
    compute_layer_sensitivity_budgets,
    evict_block_prefill,
)

_ACTIVE_EPICACHE_CACHE_REF: weakref.ReferenceType[EpiCacheCache] | None = None


def get_active_epicache_cache() -> EpiCacheCache | None:
    """Return currently active EpiCacheCache instance if alive, else None."""
    global _ACTIVE_EPICACHE_CACHE_REF
    if _ACTIVE_EPICACHE_CACHE_REF is not None:
        return _ACTIVE_EPICACHE_CACHE_REF()
    return None


def set_active_epicache_cache(cache: EpiCacheCache | None) -> None:
    """Register weak reference to active EpiCacheCache ensuring zero tensor retention leaks."""
    global _ACTIVE_EPICACHE_CACHE_REF
    if cache is not None:
        _ACTIVE_EPICACHE_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_EPICACHE_CACHE_REF = None


class EpiCacheLayer(DynamicLayer):
    """Cache layer implementing EpiCache block-prefill capping and layer-sensitivity token budgets."""

    is_sliding = False

    def __init__(
        self,
        layer_idx: int = 0,
        total_layers: int = 36,
        episode_budget_tokens: int = 1024,
        layer_sensitivity_alpha: float = 2.0,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.total_layers = int(total_layers)
        self.episode_budget_tokens = int(episode_budget_tokens)
        self.layer_sensitivity_alpha = float(layer_sensitivity_alpha)

        budgets = compute_layer_sensitivity_budgets(
            total_layers=self.total_layers,
            base_budget_per_layer=self.episode_budget_tokens,
            alpha=self.layer_sensitivity_alpha,
        )
        self.sensitivity_budget: int = budgets.get(self.layer_idx, self.episode_budget_tokens)
        # HuggingFace causal models construct a single shared 4D causal_mask using layer 0's
        # past sequence length for all full-attention layers. Keep retained KV sequence lengths
        # aligned across layers at the exact average budget (total_layers * episode_budget_tokens)
        # so multi-chunk prefill (q_len > 1) never hits cross-layer mask dimension mismatches.
        self.layer_budget: int = self.episode_budget_tokens

        self.original_positions: list[int] = []
        self.total_tokens_seen: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append incoming states and evict excess tokens on decode if exceeding layer budget."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

        new_tokens = key_states.shape[-2]
        self.original_positions.extend(
            range(self.total_tokens_seen, self.total_tokens_seen + new_tokens)
        )
        self.total_tokens_seen += new_tokens

        if new_tokens == 1 and self.keys.shape[-2] > self.layer_budget:
            self.prune()

        return self.keys, self.values

    def prune(self) -> None:
        """Prune retained block buffer to budget, preserving token 0 sink & salient keys."""
        if self.is_initialized and self.keys is not None and self.values is not None:
            seq_len = self.keys.shape[-2]
            if seq_len > self.layer_budget:
                k_ret, v_ret, retained_idx = evict_block_prefill(
                    self.keys, self.values, budget=self.layer_budget
                )
                self.keys = k_ret
                self.values = v_ret
                if self.original_positions:
                    idx_list = retained_idx.tolist()
                    self.original_positions = [
                        self.original_positions[i]
                        for i in idx_list
                        if i < len(self.original_positions)
                    ]

    def get_seq_length(self) -> int:
        return self.keys.shape[-2] if (self.is_initialized and self.keys is not None) else 0

    def get_max_length(self) -> int:
        return self.layer_budget

    def reset(self) -> None:
        self.total_tokens_seen = 0
        self.original_positions = []
        super().reset()


class EpiCacheCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing EpiCache layer-budgeted episodic storage."""

    is_epicache = True

    def __init__(
        self,
        total_layers: int = 36,
        episode_budget_tokens: int = 1024,
        layer_sensitivity_alpha: float = 2.0,
    ):
        self.total_layers = int(total_layers)
        self.episode_budget_tokens = int(episode_budget_tokens)
        self.layer_sensitivity_alpha = float(layer_sensitivity_alpha)

        super().__init__(
            layer_class_to_replicate=lambda: EpiCacheLayer(
                layer_idx=len(self.layers),
                total_layers=self.total_layers,
                episode_budget_tokens=self.episode_budget_tokens,
                layer_sensitivity_alpha=self.layer_sensitivity_alpha,
            )
        )
        set_active_epicache_cache(self)

    @property
    def original_positions(self) -> list[int]:
        if not self.layers:
            return []
        return self.layers[0].original_positions

    def prune_after_chunk(self) -> None:
        """Execute block-wise prefill pruning across all layers immediately after chunk ingestion."""
        for layer in self.layers:
            if hasattr(layer, "prune"):
                layer.prune()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self) -> int:
        if not self.layers:
            return self.episode_budget_tokens
        return self.layers[0].get_max_length()

    def __iter__(self) -> Iterator[tuple[Any, Any, Any]]:
        for layer in self.layers:
            yield getattr(layer, "keys", None), getattr(layer, "values", None), None
