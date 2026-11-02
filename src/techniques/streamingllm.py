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
_ACTIVE_STREAMING_CONFIG: dict[str, Any] = {
    "enabled": False,
    "sink_tokens": 4,
    "window_size": 2048,
    "block_size": 16,
    "num_sink_blocks": 1,
    "effective_sliding_window": 2064,
}
_HOOKS_INSTALLED: bool = False


@dataclass(frozen=True)
class CacheBoundaryInfo:
    """Token-level boundary calculation for StreamingLLM dual-region cache."""

    seq_len: int
    sink_tokens: int
    window_size: int
    sink_range: tuple[int, int]
    evicted_range: tuple[int, int]
    window_range: tuple[int, int]
    num_evicted: int
    active_tokens: int
    retained_indices: list[int]
    cache_positions: list[int]
    next_query_position: int


@dataclass(frozen=True)
class PagedBlockBoundaryInfo:
    """Block-level boundary calculation for PagedAttention KV cache."""

    num_computed_tokens: int
    block_size: int
    num_sink_blocks: int
    num_window_blocks: int
    effective_sliding_window: int
    total_blocks: int
    num_skipped_blocks: int
    evict_block_range: tuple[int, int]
    retained_sink_blocks: list[int]
    retained_window_blocks: list[int]


def compute_cache_boundaries(
    seq_len: int,
    sink_tokens: int = 4,
    window_size: int = 2048,
) -> CacheBoundaryInfo:
    """Compute StreamingLLM dual-region cache boundaries and cache-relative RoPE positions.

    Partitions a sequence of length `seq_len` into:
      1. Attention Sink region C_sink: [0, min(seq_len, sink_tokens))
      2. Evicted Middle region:        [sink_end, window_start)
      3. Rolling Window region C_rec:  [window_start, seq_len)

    Assigns contiguous cache-relative positions p_i in [0, active_tokens - 1]
    and next_query_position = active_tokens so RoPE relative distances stay
    strictly within pre-trained context bounds.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if sink_tokens < 0:
        raise ValueError(f"sink_tokens must be non-negative, got {sink_tokens}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    sink_end = min(seq_len, sink_tokens)
    sink_range = (0, sink_end)

    capacity = sink_tokens + window_size
    if seq_len <= capacity:
        evicted_range = (sink_end, sink_end)
        window_range = (sink_end, seq_len)
    else:
        window_start = seq_len - window_size
        evicted_range = (sink_end, window_start)
        window_range = (window_start, seq_len)

    num_evicted = evicted_range[1] - evicted_range[0]
    retained_indices = list(range(*sink_range)) + list(range(*window_range))
    active_tokens = len(retained_indices)
    cache_positions = list(range(active_tokens))
    next_query_position = active_tokens

    return CacheBoundaryInfo(
        seq_len=seq_len,
        sink_tokens=sink_tokens,
        window_size=window_size,
        sink_range=sink_range,
        evicted_range=evicted_range,
        window_range=window_range,
        num_evicted=num_evicted,
        active_tokens=active_tokens,
        retained_indices=retained_indices,
        cache_positions=cache_positions,
        next_query_position=next_query_position,
    )


def compute_paged_block_boundaries(
    num_computed_tokens: int,
    sink_tokens: int = 4,
    window_size: int = 2048,
    block_size: int = 16,
) -> PagedBlockBoundaryInfo:
    """Compute PagedAttention block-table boundaries for StreamingLLM.

    Pins the initial `num_sink_blocks = ceil(sink_tokens / block_size)` blocks
    permanently in GPU VRAM while evicting middle blocks `[num_sink_blocks, num_skipped_blocks)`
    outside the rolling local window.
    """
    if num_computed_tokens < 0:
        raise ValueError(f"num_computed_tokens must be >= 0, got {num_computed_tokens}")
    if sink_tokens < 0:
        raise ValueError(f"sink_tokens must be >= 0, got {sink_tokens}")
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    num_sink_blocks = math.ceil(sink_tokens / block_size) if sink_tokens > 0 else 0
    num_window_blocks = math.ceil(window_size / block_size)
    effective_sliding_window = num_sink_blocks * block_size + window_size

    total_blocks = math.ceil(num_computed_tokens / block_size) if num_computed_tokens > 0 else 0
    num_skipped_tokens = max(0, num_computed_tokens - effective_sliding_window + 1)
    num_skipped_blocks = min(num_skipped_tokens // block_size, total_blocks)

    evict_start = min(num_sink_blocks, num_skipped_blocks)
    evict_end = max(evict_start, num_skipped_blocks)
    evict_block_range = (evict_start, evict_end)

    retained_sink_blocks = list(range(0, min(num_sink_blocks, total_blocks)))
    window_block_start = max(min(num_sink_blocks, total_blocks), evict_end)
    retained_window_blocks = list(range(window_block_start, total_blocks))

    return PagedBlockBoundaryInfo(
        num_computed_tokens=num_computed_tokens,
        block_size=block_size,
        num_sink_blocks=num_sink_blocks,
        num_window_blocks=num_window_blocks,
        effective_sliding_window=effective_sliding_window,
        total_blocks=total_blocks,
        num_skipped_blocks=num_skipped_blocks,
        evict_block_range=evict_block_range,
        retained_sink_blocks=retained_sink_blocks,
        retained_window_blocks=retained_window_blocks,
    )


def evict_streaming_kv_cache(
    key_cache: "torch.Tensor",
    value_cache: "torch.Tensor",
    sink_tokens: int = 4,
    window_size: int = 2048,
    seq_dim: int = -2,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Prune middle tokens from Key and Value cache tensors to retain C_sink + C_recent.

    Supports both 4D `(batch, num_kv_heads, seq_len, head_dim)` and 3D
    `(seq_len, num_kv_heads, head_dim)` tensor layouts while strictly preserving
    the original floating-point dtype (BF16/FP16).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for evict_streaming_kv_cache")
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key_cache shape {key_cache.shape} must match value_cache shape {value_cache.shape}"
        )

    seq_len = key_cache.size(seq_dim)
    capacity = sink_tokens + window_size
    if seq_len <= capacity:
        return key_cache, value_cache

    parts_k: list[torch.Tensor] = []
    parts_v: list[torch.Tensor] = []
    if sink_tokens > 0:
        parts_k.append(key_cache.narrow(seq_dim, 0, sink_tokens))
        parts_v.append(value_cache.narrow(seq_dim, 0, sink_tokens))

    window_start = seq_len - window_size
    parts_k.append(key_cache.narrow(seq_dim, window_start, window_size))
    parts_v.append(value_cache.narrow(seq_dim, window_start, window_size))

    return torch.cat(parts_k, dim=seq_dim), torch.cat(parts_v, dim=seq_dim)


def compute_gqa_streaming_attention(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    sink_tokens: int = 4,
    window_size: int = 2048,
    scale: float | None = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Compute Grouped-Query Attention (GQA) over a StreamingLLM pruned KV cache.

    Args:
        query: Tensor of shape `(batch, num_q_heads, q_len, head_dim)`
        key:   Tensor of shape `(batch, num_kv_heads, kv_len, head_dim)`
        value: Tensor of shape `(batch, num_kv_heads, kv_len, head_dim)`
        sink_tokens: Number of initial attention sink tokens to anchor ($N_{sink}$).
        window_size: Rolling local window size ($W$).
        scale: Softmax scaling factor (defaults to `1 / sqrt(head_dim)`).

    Returns:
        Tuple of `(attn_output, attn_weights)` where:
          - `attn_output` has shape `(batch, num_q_heads, q_len, head_dim)`
          - `attn_weights` has shape `(batch, num_q_heads, q_len, active_kv_len)`
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for compute_gqa_streaming_attention")

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be 4D tensors (B, H, L, D)")

    num_q_heads = query.size(1)
    num_kv_heads = key.size(1)
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    pruned_k, pruned_v = evict_streaming_kv_cache(
        key, value, sink_tokens=sink_tokens, window_size=window_size, seq_dim=2
    )

    group_size = num_q_heads // num_kv_heads
    if group_size > 1:
        expanded_k = pruned_k.repeat_interleave(group_size, dim=1)
        expanded_v = pruned_v.repeat_interleave(group_size, dim=1)
    else:
        expanded_k = pruned_k
        expanded_v = pruned_v

    head_dim = query.size(-1)
    softmax_scale = scale if scale is not None else (head_dim ** -0.5)

    scores = torch.matmul(query.float(), expanded_k.float().transpose(-2, -1)) * softmax_scale
    attn_weights = F.softmax(scores, dim=-1).to(query.dtype)
    attn_output = torch.matmul(attn_weights, expanded_v)
    return attn_output, attn_weights


def _install_vllm_streamingllm_hooks() -> None:
    """Install idempotent runtime hooks into vLLM V1 KV cache manager and FA builder."""
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

    # 1. Ensure dense GQA models (like Qwen3) propagate hf_overrides sliding_window
    # across all layers when StreamingLLM is active.
    orig_get_sliding_window = vllm_model_cfg.ModelConfig.get_sliding_window

    def _streaming_get_sliding_window(self) -> int | None:
        if _ACTIVE_STREAMING_CONFIG.get("enabled", False):
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

    vllm_model_cfg.ModelConfig.get_sliding_window = _streaming_get_sliding_window  # type: ignore[method-assign]

    # 2. Reserve extra admission blocks in SlidingWindowSpec for pinned sink blocks.
    orig_max_admission = vllm_kv_iface.SlidingWindowSpec.max_admission_blocks_per_request

    def _streaming_max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        base_blocks = orig_max_admission(self, max_in_flight_tokens, max_model_len)
        if _ACTIVE_STREAMING_CONFIG.get("enabled", False):
            num_sink_blocks = int(_ACTIVE_STREAMING_CONFIG.get("num_sink_blocks", 1))
            return base_blocks + num_sink_blocks
        return base_blocks

    vllm_kv_iface.SlidingWindowSpec.max_admission_blocks_per_request = (  # type: ignore[method-assign]
        _streaming_max_admission_blocks_per_request
    )

    # 3. Pin initial sink blocks [0, num_sink_blocks) in SlidingWindowManager
    # while freeing middle blocks [num_sink_blocks, num_skipped_blocks).
    orig_remove_skipped_blocks = vllm_kv_mgr.SlidingWindowManager.remove_skipped_blocks

    def _streaming_remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        if not _ACTIVE_STREAMING_CONFIG.get("enabled", False):
            orig_remove_skipped_blocks(
                self, request_id, processed_computed_tokens, num_prompt_tokens
            )
            return

        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
        if num_skipped_tokens <= 0:
            return
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return
        sink_tokens = int(_ACTIVE_STREAMING_CONFIG.get("sink_tokens", 4))
        num_sink_blocks = math.ceil(sink_tokens / self.block_size) if sink_tokens > 0 else 0
        num_skipped_blocks = min(num_skipped_tokens // self.block_size, len(blocks))
        if num_skipped_blocks > num_sink_blocks:
            self._remove_blocks_in_range(request_id, num_sink_blocks, num_skipped_blocks)

    vllm_kv_mgr.SlidingWindowManager.remove_skipped_blocks = (  # type: ignore[method-assign]
        _streaming_remove_skipped_blocks
    )

    # 4. Stage pinned sink block IDs into the leading slots of the active sliding
    # window in FlashAttentionMetadataBuilder so FlashAttention attends to C_sink + C_recent.
    orig_fa_build = vllm_fa.FlashAttentionMetadataBuilder.build

    def _streaming_fa_build(
        self,
        common_prefix_len: int,
        common_attn_metadata: Any,
        fast_build: bool = False,
    ) -> Any:
        if _ACTIVE_STREAMING_CONFIG.get("enabled", False) and torch is not None:
            eff_window = int(_ACTIVE_STREAMING_CONFIG.get("effective_sliding_window", 2064))
            num_sink_blocks = int(_ACTIVE_STREAMING_CONFIG.get("num_sink_blocks", 1))
            block_size = int(getattr(self, "block_size", _ACTIVE_STREAMING_CONFIG.get("block_size", 16)))
            seq_lens = getattr(common_attn_metadata, "seq_lens", None)
            blk_table = getattr(common_attn_metadata, "block_table_tensor", None)

            if (
                num_sink_blocks > 0
                and seq_lens is not None
                and blk_table is not None
                and blk_table.ndim == 2
                and blk_table.size(1) > num_sink_blocks
            ):
                num_reqs = min(int(common_attn_metadata.num_reqs), blk_table.size(0))
                seq_lens_list = seq_lens[:num_reqs].tolist()
                if any(s > eff_window for s in seq_lens_list):
                    staged_table = blk_table.clone()
                    max_cols = staged_table.size(1)
                    for req_idx, s_len in enumerate(seq_lens_list):
                        if s_len > eff_window:
                            win_start_tok = s_len - eff_window
                            win_start_blk = win_start_tok // block_size
                            if (
                                win_start_blk >= num_sink_blocks
                                and win_start_blk + num_sink_blocks <= max_cols
                            ):
                                staged_table[
                                    req_idx, win_start_blk : win_start_blk + num_sink_blocks
                                ] = staged_table[req_idx, :num_sink_blocks]
                    common_attn_metadata.block_table_tensor = staged_table

        return orig_fa_build(self, common_prefix_len, common_attn_metadata, fast_build)

    vllm_fa.FlashAttentionMetadataBuilder.build = _streaming_fa_build  # type: ignore[method-assign]
    _HOOKS_INSTALLED = True


@register("streamingllm")
class StreamingLLMTechnique(Technique):
    """StreamingLLM: Attention sink anchoring + rolling local sliding window.

    Pins the initial $N_{\text{sink}}$ tokens (first PagedAttention block) in GPU
    VRAM while maintaining a rolling local sliding window of $W$ tokens and
    evicting middle tokens. Preserves Softmax attention sink denominators across
    all grouped query heads in Dense GQA architectures without modifying BF16
    base weights.

    Axes:
        Axis.STRUCTURAL: Prunes middle tokens from the active KV cache representation.
    """

    axes = Axis.STRUCTURAL

    def __init__(
        self,
        sink_tokens: int = 4,
        window_size: int = 2048,
        block_size: int = 16,
    ):
        if int(sink_tokens) < 0:
            raise ValueError(f"sink_tokens must be >= 0, got {sink_tokens}")
        if int(window_size) <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if int(block_size) <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        self.sink_tokens = int(sink_tokens)
        self.window_size = int(window_size)
        self.block_size = int(block_size)

    @property
    def num_sink_blocks(self) -> int:
        return math.ceil(self.sink_tokens / self.block_size) if self.sink_tokens > 0 else 0

    @property
    def effective_cache_tokens(self) -> int:
        return self.sink_tokens + self.window_size

    @property
    def effective_sliding_window(self) -> int:
        """Block-aligned sliding window spanning pinned sink blocks + rolling window."""
        return self.num_sink_blocks * self.block_size + self.window_size

    def compute_boundaries(self, seq_len: int) -> CacheBoundaryInfo:
        """Compute token-level dual-region cache boundaries for `seq_len`."""
        return compute_cache_boundaries(
            seq_len=seq_len,
            sink_tokens=self.sink_tokens,
            window_size=self.window_size,
        )

    def compute_block_boundaries(self, num_computed_tokens: int) -> PagedBlockBoundaryInfo:
        """Compute PagedAttention block-level boundaries for `num_computed_tokens`."""
        return compute_paged_block_boundaries(
            num_computed_tokens=num_computed_tokens,
            sink_tokens=self.sink_tokens,
            window_size=self.window_size,
            block_size=self.block_size,
        )

    def setup(self) -> None:
        """Configure active StreamingLLM state and install process-level vLLM hooks."""
        _ACTIVE_STREAMING_CONFIG.update(
            {
                "enabled": True,
                "sink_tokens": self.sink_tokens,
                "window_size": self.window_size,
                "block_size": self.block_size,
                "num_sink_blocks": self.num_sink_blocks,
                "effective_sliding_window": self.effective_sliding_window,
            }
        )
        _install_vllm_streamingllm_hooks()

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        """Return valid vllm.EngineArgs kwargs configuring block size and sliding window."""
        del context_length, max_tokens
        return {
            "block_size": self.block_size,
            "enable_prefix_caching": False,
            "hf_overrides": {
                "use_sliding_window": True,
                "sliding_window": self.effective_sliding_window,
                "max_window_layers": 0,
            },
        }

    def workload(self) -> Workload:
        return SyntheticFiller()

    def pytorch_cache_factory(self) -> Any:
        from src.techniques.streaming_cache import StreamingCache

        return lambda: StreamingCache(
            sink_tokens=self.sink_tokens,
            window_size=self.window_size,
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
            "sink_tokens": self.sink_tokens,
            "window_size": self.window_size,
            "block_size": self.block_size,
            "num_sink_blocks": self.num_sink_blocks,
            "effective_cache_tokens": self.effective_cache_tokens,
            "effective_sliding_window": self.effective_sliding_window,
            "active_kv_tokens": boundary.active_tokens,
            "evicted_middle_tokens": boundary.num_evicted,
            "evicted_middle_blocks": (
                block_boundary.evict_block_range[1] - block_boundary.evict_block_range[0]
            ),
        }
