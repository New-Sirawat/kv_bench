from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from src.core.axis import Axis
from src.core.registry import register
from src.core.technique import Technique
from src.core.workload import SyntheticFiller, Workload

# Process-level runtime configuration read by vLLM hooks installed in setup().
_ACTIVE_H2O_CONFIG: dict[str, Any] = {
    "enabled": False,
    "heavy_hitter_tokens": 1024,
    "recent_window_tokens": 1024,
    "block_size": 16,
    "effective_cache_tokens": 2048,
    "num_hh_blocks": 64,
    "num_recent_blocks": 64,
}
_HOOKS_INSTALLED: bool = False


@dataclass(frozen=True)
class H2OBoundaryInfo:
    """Token-level boundary and index calculation for H2O (Heavy-Hitter Oracle) cache."""

    seq_len: int
    heavy_hitter_tokens: int
    recent_window_tokens: int
    total_budget: int
    retained_hh_indices: list[int]
    retained_recent_indices: list[int]
    retained_indices: list[int]
    num_evicted: int
    active_tokens: int
    cache_positions: list[int]
    next_query_position: int


@dataclass(frozen=True)
class H2OPagedBlockInfo:
    """Block-level boundary calculation for PagedAttention H2O KV cache."""

    num_computed_tokens: int
    block_size: int
    num_hh_blocks: int
    num_recent_blocks: int
    effective_cache_blocks: int
    total_blocks: int
    retained_hh_blocks: list[int]
    retained_recent_blocks: list[int]
    retained_blocks: list[int]
    evicted_blocks: list[int]


def compute_h2o_boundaries(
    seq_len: int,
    cumulative_scores: list[float] | "torch.Tensor" | None = None,
    heavy_hitter_tokens: int = 1024,
    recent_window_tokens: int = 1024,
) -> H2OBoundaryInfo:
    """Compute H2O token retention indices based on cumulative attention scores.

    Partitions a sequence of length `seq_len` into:
      1. Recent window region C_recent: [max(0, seq_len - recent_window_tokens), seq_len)
         which is always retained to protect local syntax and context.
      2. Candidate region for Heavy Hitters: [0, max(0, seq_len - recent_window_tokens))
         from which the top `heavy_hitter_tokens` with highest cumulative attention
         scores are selected (C_hh).
      3. Evicted tokens: Candidates outside C_hh with lowest cumulative attention.

    When `seq_len <= heavy_hitter_tokens + recent_window_tokens`, all tokens are retained.
    Retained indices are returned in strictly ascending chronological order.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if heavy_hitter_tokens < 0:
        raise ValueError(f"heavy_hitter_tokens must be non-negative, got {heavy_hitter_tokens}")
    if recent_window_tokens <= 0:
        raise ValueError(f"recent_window_tokens must be positive, got {recent_window_tokens}")

    total_budget = heavy_hitter_tokens + recent_window_tokens

    if seq_len <= total_budget:
        retained_indices = list(range(seq_len))
        recent_start = max(0, seq_len - recent_window_tokens)
        retained_hh = list(range(0, recent_start))
        retained_recent = list(range(recent_start, seq_len))
        return H2OBoundaryInfo(
            seq_len=seq_len,
            heavy_hitter_tokens=heavy_hitter_tokens,
            recent_window_tokens=recent_window_tokens,
            total_budget=total_budget,
            retained_hh_indices=retained_hh,
            retained_recent_indices=retained_recent,
            retained_indices=retained_indices,
            num_evicted=0,
            active_tokens=seq_len,
            cache_positions=list(range(seq_len)),
            next_query_position=seq_len,
        )

    # Sequence exceeds capacity budget -> evict low-score candidate tokens
    recent_start = seq_len - recent_window_tokens
    retained_recent = list(range(recent_start, seq_len))
    candidate_indices = list(range(0, recent_start))

    if heavy_hitter_tokens == 0:
        retained_hh = []
    elif cumulative_scores is None:
        # Graceful fallback: retain first heavy_hitter_tokens (prefix/sink behavior)
        retained_hh = candidate_indices[:heavy_hitter_tokens]
    else:
        # Extract 1D score vector for candidates
        if torch is not None and isinstance(cumulative_scores, torch.Tensor):
            scores_tensor = cumulative_scores
            if scores_tensor.ndim > 1:
                # Average across batch and heads if multidimensional
                dim_to_reduce = tuple(range(scores_tensor.ndim - 1))
                scores_tensor = scores_tensor.mean(dim=dim_to_reduce)
            cand_scores = scores_tensor[:recent_start].tolist()
        else:
            cand_scores = list(cumulative_scores)[:recent_start]

        # Select top-k heavy hitter indices
        k = min(heavy_hitter_tokens, len(candidate_indices))
        # Sort candidate indices by score descending, breaking ties by index
        indexed_scores = [(cand_scores[i] if i < len(cand_scores) else 0.0, i) for i in candidate_indices]
        top_k_candidates = sorted(indexed_scores, key=lambda x: x[0], reverse=True)[:k]
        retained_hh = sorted([item[1] for item in top_k_candidates])

    retained_indices = sorted(retained_hh + retained_recent)
    active_tokens = len(retained_indices)
    num_evicted = seq_len - active_tokens

    return H2OBoundaryInfo(
        seq_len=seq_len,
        heavy_hitter_tokens=heavy_hitter_tokens,
        recent_window_tokens=recent_window_tokens,
        total_budget=total_budget,
        retained_hh_indices=retained_hh,
        retained_recent_indices=retained_recent,
        retained_indices=retained_indices,
        num_evicted=num_evicted,
        active_tokens=active_tokens,
        cache_positions=list(range(active_tokens)),
        next_query_position=active_tokens,
    )


def compute_h2o_paged_block_boundaries(
    num_computed_tokens: int,
    heavy_hitter_tokens: int = 1024,
    recent_window_tokens: int = 1024,
    block_size: int = 16,
    block_scores: list[float] | None = None,
) -> H2OPagedBlockInfo:
    """Compute PagedAttention block-table retention and eviction for H2O.

    Partitions total blocks into:
      1. Recent window blocks: ceil(recent_window_tokens / block_size) blocks at the tail.
      2. Heavy hitter blocks: ceil(heavy_hitter_tokens / block_size) blocks selected
         from candidate blocks outside the recent window based on cumulative attention scores.
      3. Evicted blocks: candidate blocks not selected for heavy hitter retention.
    """
    if num_computed_tokens < 0:
        raise ValueError(f"num_computed_tokens must be non-negative, got {num_computed_tokens}")
    if heavy_hitter_tokens < 0:
        raise ValueError(f"heavy_hitter_tokens must be non-negative, got {heavy_hitter_tokens}")
    if recent_window_tokens <= 0:
        raise ValueError(f"recent_window_tokens must be positive, got {recent_window_tokens}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    num_hh_blocks = math.ceil(heavy_hitter_tokens / block_size) if heavy_hitter_tokens > 0 else 0
    num_recent_blocks = math.ceil(recent_window_tokens / block_size)
    effective_cache_blocks = num_hh_blocks + num_recent_blocks

    total_blocks = math.ceil(num_computed_tokens / block_size) if num_computed_tokens > 0 else 0

    if total_blocks <= effective_cache_blocks:
        recent_start_block = max(0, total_blocks - num_recent_blocks)
        retained_hh = list(range(0, recent_start_block))
        retained_recent = list(range(recent_start_block, total_blocks))
        retained_blocks = list(range(total_blocks))
        return H2OPagedBlockInfo(
            num_computed_tokens=num_computed_tokens,
            block_size=block_size,
            num_hh_blocks=num_hh_blocks,
            num_recent_blocks=num_recent_blocks,
            effective_cache_blocks=effective_cache_blocks,
            total_blocks=total_blocks,
            retained_hh_blocks=retained_hh,
            retained_recent_blocks=retained_recent,
            retained_blocks=retained_blocks,
            evicted_blocks=[],
        )

    recent_start_blk = total_blocks - num_recent_blocks
    retained_recent = list(range(recent_start_blk, total_blocks))
    candidate_blks = list(range(0, recent_start_blk))

    if num_hh_blocks == 0:
        retained_hh = []
    elif block_scores is None:
        retained_hh = candidate_blks[:num_hh_blocks]
    else:
        k = min(num_hh_blocks, len(candidate_blks))
        indexed = [(block_scores[i] if i < len(block_scores) else 0.0, i) for i in candidate_blks]
        top_k = sorted(indexed, key=lambda x: x[0], reverse=True)[:k]
        retained_hh = sorted([item[1] for item in top_k])

    retained_blocks = sorted(retained_hh + retained_recent)
    evicted_blocks = sorted(list(set(candidate_blks) - set(retained_hh)))

    return H2OPagedBlockInfo(
        num_computed_tokens=num_computed_tokens,
        block_size=block_size,
        num_hh_blocks=num_hh_blocks,
        num_recent_blocks=num_recent_blocks,
        effective_cache_blocks=effective_cache_blocks,
        total_blocks=total_blocks,
        retained_hh_blocks=retained_hh,
        retained_recent_blocks=retained_recent,
        retained_blocks=retained_blocks,
        evicted_blocks=evicted_blocks,
    )


def evict_h2o_kv_cache(
    key_cache: "torch.Tensor",
    value_cache: "torch.Tensor",
    retained_indices: list[int] | "torch.Tensor",
    seq_dim: int = -2,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Prune unselected tokens from Key and Value cache tensors to retain only C_hh + C_recent.

    Supports both 4D `(batch, num_kv_heads, seq_len, head_dim)` and 3D
    `(seq_len, num_kv_heads, head_dim)` tensor layouts while strictly preserving
    the original floating-point dtype (BF16/FP16).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for evict_h2o_kv_cache")
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key_cache shape {key_cache.shape} must match value_cache shape {value_cache.shape}"
        )

    seq_len = key_cache.size(seq_dim)
    if isinstance(retained_indices, torch.Tensor):
        indices_tensor = retained_indices.to(device=key_cache.device, dtype=torch.long)
    else:
        indices_tensor = torch.tensor(retained_indices, device=key_cache.device, dtype=torch.long)

    if indices_tensor.numel() == seq_len:
        return key_cache, value_cache

    pruned_k = torch.index_select(key_cache, dim=seq_dim, index=indices_tensor)
    pruned_v = torch.index_select(value_cache, dim=seq_dim, index=indices_tensor)
    return pruned_k, pruned_v


def compute_gqa_h2o_attention(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    cumulative_scores: "torch.Tensor | None" = None,
    heavy_hitter_tokens: int = 1024,
    recent_window_tokens: int = 1024,
    scale: float | None = None,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Compute Grouped-Query Attention (GQA) with H2O cumulative score tracking and eviction.

    Pools attention weights across grouped query heads into corresponding KV heads:
      score_pool = mean_{q in group}(attn_weights)
    Accumulates scores over decoding steps and applies H2O budget pruning when
    sequence length exceeds `heavy_hitter_tokens + recent_window_tokens`.

    Args:
        query: Tensor of shape `(batch, num_q_heads, q_len, head_dim)`
        key:   Tensor of shape `(batch, num_kv_heads, kv_len, head_dim)`
        value: Tensor of shape `(batch, num_kv_heads, kv_len, head_dim)`
        cumulative_scores: Optional tensor of shape `(batch, num_kv_heads, kv_len)`
        heavy_hitter_tokens: Number of heavy hitter tokens to retain ($k_{hh}$).
        recent_window_tokens: Rolling recent window tokens to retain ($k_{recent}$).
        scale: Softmax scaling factor (defaults to `1 / sqrt(head_dim)`).

    Returns:
        Tuple of `(attn_output, attn_weights, updated_cumulative_scores)` where:
          - `attn_output` has shape `(batch, num_q_heads, q_len, head_dim)`
          - `attn_weights` has shape `(batch, num_q_heads, q_len, active_kv_len)`
          - `updated_cumulative_scores` has shape `(batch, num_kv_heads, active_kv_len)`
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for compute_gqa_h2o_attention")

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be 4D tensors (B, H, L, D)")

    batch_size = query.size(0)
    num_q_heads = query.size(1)
    num_kv_heads = key.size(1)
    kv_len = key.size(2)

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    group_size = num_q_heads // num_kv_heads
    if group_size > 1:
        expanded_k = key.repeat_interleave(group_size, dim=1)
        expanded_v = value.repeat_interleave(group_size, dim=1)
    else:
        expanded_k = key
        expanded_v = value

    head_dim = query.size(-1)
    softmax_scale = scale if scale is not None else (head_dim ** -0.5)

    scores = torch.matmul(query.float(), expanded_k.float().transpose(-2, -1)) * softmax_scale
    attn_weights = F.softmax(scores, dim=-1).to(query.dtype)
    attn_output = torch.matmul(attn_weights, expanded_v)

    # Pool attention across query heads within each KV head group:
    # shape: (batch, num_kv_heads, group_size, q_len, kv_len) -> mean over group_size -> sum over q_len
    pooled_step_scores = attn_weights.view(
        batch_size, num_kv_heads, group_size, query.size(2), kv_len
    ).mean(dim=2).sum(dim=2)  # shape: (batch, num_kv_heads, kv_len)

    if cumulative_scores is not None:
        if cumulative_scores.shape == pooled_step_scores.shape:
            updated_scores = cumulative_scores + pooled_step_scores
        else:
            # Broadcast or adjust if shapes differ
            updated_scores = pooled_step_scores
    else:
        updated_scores = pooled_step_scores

    capacity = heavy_hitter_tokens + recent_window_tokens
    if kv_len <= capacity:
        return attn_output, attn_weights, updated_scores

    # Prune KV cache and updated_scores using pooled sequence score
    seq_scores = updated_scores.mean(dim=(0, 1))  # (kv_len,)
    boundary = compute_h2o_boundaries(
        seq_len=kv_len,
        cumulative_scores=seq_scores,
        heavy_hitter_tokens=heavy_hitter_tokens,
        recent_window_tokens=recent_window_tokens,
    )

    retained_tensor = torch.tensor(
        boundary.retained_indices, device=key.device, dtype=torch.long
    )
    pruned_scores = torch.index_select(updated_scores, dim=-1, index=retained_tensor)

    return attn_output, attn_weights, pruned_scores


def _install_vllm_h2o_hooks() -> None:
    """Install idempotent runtime hooks into vLLM V1 KV cache manager for H2O."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    try:
        import vllm.config.model as vllm_model_cfg
        import vllm.v1.attention.backends.flash_attn as vllm_fa
        import vllm.v1.core.single_type_kv_cache_manager as vllm_kv_mgr
        import vllm.v1.kv_cache_interface as vllm_kv_iface
    except ImportError:
        return

    # Ensure dense GQA models (like Qwen3) propagate sliding_window across all layers when H2O is active
    orig_get_sliding_window = vllm_model_cfg.ModelConfig.get_sliding_window

    def _h2o_get_sliding_window(self) -> int | None:
        if _ACTIVE_H2O_CONFIG.get("enabled", False):
            hf_text_cfg = getattr(self, "hf_text_config", None)
            if hf_text_cfg is not None and getattr(hf_text_cfg, "use_sliding_window", False):
                if getattr(hf_text_cfg, "max_window_layers", None) == 0:
                    num_layers = getattr(hf_text_cfg, "num_hidden_layers", 0)
                    if num_layers > 0 and hasattr(hf_text_cfg, "layer_types"):
                        hf_text_cfg.layer_types = ["sliding_attention"] * num_layers
                sw = getattr(hf_text_cfg, "sliding_window", None)
                if sw is not None:
                    return int(sw)
        return orig_get_sliding_window(self)

    vllm_model_cfg.ModelConfig.get_sliding_window = _h2o_get_sliding_window  # type: ignore[method-assign]

    # Reserve admission blocks for heavy hitters and recent window in SlidingWindowSpec
    orig_max_admission = vllm_kv_iface.SlidingWindowSpec.max_admission_blocks_per_request

    def _h2o_max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        base_blocks = orig_max_admission(self, max_in_flight_tokens, max_model_len)
        if _ACTIVE_H2O_CONFIG.get("enabled", False):
            num_hh_blocks = int(_ACTIVE_H2O_CONFIG.get("num_hh_blocks", 64))
            return base_blocks + num_hh_blocks
        return base_blocks

    vllm_kv_iface.SlidingWindowSpec.max_admission_blocks_per_request = (  # type: ignore[method-assign]
        _h2o_max_admission_blocks_per_request
    )

    _HOOKS_INSTALLED = True


_PYTORCH_HOOKS_INSTALLED: bool = False
_ORIG_SDPA: Any = None


def _install_pytorch_h2o_hooks() -> None:
    """Install attention wrapper into transformers ALL_ATTENTION_FUNCTIONS for H2O."""
    global _PYTORCH_HOOKS_INSTALLED, _ORIG_SDPA
    if _PYTORCH_HOOKS_INSTALLED:
        return

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from src.techniques.h2o_cache import get_active_h2o_cache, record_h2o_attention_scores
    except ImportError:
        return

    orig_sdpa = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
    if orig_sdpa is None:
        return

    _ORIG_SDPA = orig_sdpa

    def h2o_sdpa_attention_forward(module, query, key, value, *args, **kwargs):
        output, attn_weights = _ORIG_SDPA(module, query, key, value, *args, **kwargs)

        cache = get_active_h2o_cache()
        if cache is not None and getattr(cache, "is_h2o", False):
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None and layer_idx < len(cache.layers):
                layer = cache.layers[layer_idx]
                record_h2o_attention_scores(layer, query, key)
                # In decode step (single token query), prune immediately if over capacity
                if query.shape[-2] == 1:
                    layer.prune()

        return output, attn_weights

    ALL_ATTENTION_FUNCTIONS.register("sdpa", h2o_sdpa_attention_forward)
    _PYTORCH_HOOKS_INSTALLED = True


def _uninstall_pytorch_h2o_hooks() -> None:
    """Restore original sdpa attention forward in transformers ALL_ATTENTION_FUNCTIONS."""
    global _PYTORCH_HOOKS_INSTALLED, _ORIG_SDPA
    if not _PYTORCH_HOOKS_INSTALLED or _ORIG_SDPA is None:
        return
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS.register("sdpa", _ORIG_SDPA)
        _PYTORCH_HOOKS_INSTALLED = False
        _ORIG_SDPA = None
    except ImportError:
        pass


@register("h2o")
class H2OTechnique(Technique):
    """H2O: Heavy-Hitter Oracle KV-cache eviction.

    Dynamically tracks cumulative attention scores across all tokens and retains
    only the top-$k_{hh}$ heavy hitter tokens alongside a rolling recent window of
    $k_{recent}$ tokens (total budget = $k_{hh} + k_{recent}$). Prunes less
    frequently attended tokens from the active KV cache representation while preserving
    dense GQA grouped query head stability and leaving base BF16 weights unmodified.

    Axes:
        Axis.STRUCTURAL: Prunes low-attention tokens from the active KV cache representation.
    """

    axes = Axis.STRUCTURAL

    def __init__(
        self,
        heavy_hitter_tokens: int = 1024,
        recent_window_tokens: int = 1024,
        block_size: int = 16,
    ):
        if int(heavy_hitter_tokens) < 0:
            raise ValueError(f"heavy_hitter_tokens must be >= 0, got {heavy_hitter_tokens}")
        if int(recent_window_tokens) <= 0:
            raise ValueError(f"recent_window_tokens must be > 0, got {recent_window_tokens}")
        if int(block_size) <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        self.heavy_hitter_tokens = int(heavy_hitter_tokens)
        self.recent_window_tokens = int(recent_window_tokens)
        self.block_size = int(block_size)

    @property
    def effective_cache_tokens(self) -> int:
        """Total active KV cache capacity budget (heavy hitters + recent window)."""
        return self.heavy_hitter_tokens + self.recent_window_tokens

    @property
    def num_hh_blocks(self) -> int:
        """Number of PagedAttention blocks dedicated to heavy hitters."""
        return math.ceil(self.heavy_hitter_tokens / self.block_size) if self.heavy_hitter_tokens > 0 else 0

    @property
    def num_recent_blocks(self) -> int:
        """Number of PagedAttention blocks dedicated to recent local window."""
        return math.ceil(self.recent_window_tokens / self.block_size)

    @property
    def effective_cache_blocks(self) -> int:
        """Total PagedAttention blocks retained in cache."""
        return self.num_hh_blocks + self.num_recent_blocks

    def compute_boundaries(
        self,
        seq_len: int,
        cumulative_scores: list[float] | "torch.Tensor" | None = None,
    ) -> H2OBoundaryInfo:
        """Compute token-level H2O retention and eviction boundaries for `seq_len`."""
        return compute_h2o_boundaries(
            seq_len=seq_len,
            cumulative_scores=cumulative_scores,
            heavy_hitter_tokens=self.heavy_hitter_tokens,
            recent_window_tokens=self.recent_window_tokens,
        )

    def compute_block_boundaries(
        self,
        num_computed_tokens: int,
        block_scores: list[float] | None = None,
    ) -> H2OPagedBlockInfo:
        """Compute PagedAttention block-level boundaries for `num_computed_tokens`."""
        return compute_h2o_paged_block_boundaries(
            num_computed_tokens=num_computed_tokens,
            heavy_hitter_tokens=self.heavy_hitter_tokens,
            recent_window_tokens=self.recent_window_tokens,
            block_size=self.block_size,
            block_scores=block_scores,
        )

    def setup(self) -> None:
        """Configure active H2O state and install process-level vLLM and PyTorch hooks."""
        _ACTIVE_H2O_CONFIG.update(
            {
                "enabled": True,
                "heavy_hitter_tokens": self.heavy_hitter_tokens,
                "recent_window_tokens": self.recent_window_tokens,
                "block_size": self.block_size,
                "effective_cache_tokens": self.effective_cache_tokens,
                "num_hh_blocks": self.num_hh_blocks,
                "num_recent_blocks": self.num_recent_blocks,
            }
        )
        _install_vllm_h2o_hooks()
        _install_pytorch_h2o_hooks()

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        """Return valid vllm.EngineArgs kwargs configuring block size and sliding window."""
        del context_length, max_tokens
        return {
            "block_size": self.block_size,
            "enable_prefix_caching": False,
            "hf_overrides": {
                "use_sliding_window": True,
                "sliding_window": self.effective_cache_tokens,
                "max_window_layers": 0,
            },
        }

    def workload(self) -> Workload:
        return SyntheticFiller()

    def pytorch_cache_factory(self) -> Any:
        from src.techniques.h2o_cache import H2OCache

        return lambda: H2OCache(
            heavy_hitter_tokens=self.heavy_hitter_tokens,
            recent_window_tokens=self.recent_window_tokens,
        )

    def extra_metrics(self, llm, workload_output) -> dict[str, Any]:
        del llm
        total_seq_len = 0
        if workload_output is not None:
            total_seq_len = int(getattr(workload_output, "prompt_tokens", 0)) + int(
                getattr(workload_output, "generated_tokens", 0)
            )
        boundary = self.compute_boundaries(total_seq_len)
        block_boundary = self.compute_block_boundaries(total_seq_len)
        return {
            "heavy_hitter_tokens": self.heavy_hitter_tokens,
            "recent_window_tokens": self.recent_window_tokens,
            "block_size": self.block_size,
            "num_hh_blocks": self.num_hh_blocks,
            "num_recent_blocks": self.num_recent_blocks,
            "effective_cache_tokens": self.effective_cache_tokens,
            "effective_cache_blocks": self.effective_cache_blocks,
            "active_kv_tokens": boundary.active_tokens,
            "evicted_tokens": boundary.num_evicted,
            "evicted_blocks": len(block_boundary.evicted_blocks),
        }
