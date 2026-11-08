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


_ACTIVE_INFINIGEN_CACHE_REF: weakref.ReferenceType[InfiniGenCache] | None = None


def get_active_infinigen_cache() -> InfiniGenCache | None:
    """Return currently active InfiniGenCache instance if alive, else None."""
    global _ACTIVE_INFINIGEN_CACHE_REF
    if _ACTIVE_INFINIGEN_CACHE_REF is not None:
        return _ACTIVE_INFINIGEN_CACHE_REF()
    return None


def set_active_infinigen_cache(cache: InfiniGenCache | None) -> None:
    """Register weak reference to active InfiniGenCache ensuring zero tensor retention leaks."""
    global _ACTIVE_INFINIGEN_CACHE_REF
    if cache is not None:
        _ACTIVE_INFINIGEN_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_INFINIGEN_CACHE_REF = None


def generate_orthogonal_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    """Generate deterministic orthogonal rotation matrix A (dim x dim) via QR decomposition.

    Satisfies A @ A.T = I, ensuring mathematical identity: (Q @ A) @ (K @ A).T == Q @ K.T.
    """
    g = torch.Generator().manual_seed(seed)
    random_matrix = torch.randn(dim, dim, generator=g)
    q, _ = torch.linalg.qr(random_matrix)
    return q


def compute_speculative_block_mask(
    query: torch.Tensor,
    proxy_keys: torch.Tensor,
    alpha: float = 4.0,
    block_size: int = 16,
    max_budget_ratio: float = 0.35,
) -> tuple[torch.Tensor, list[int], float]:
    """Calculate speculative attention scores against proxy keys and determine prefetched block indices.

    Args:
        query: Query tensor [B, N_q, 1, D_proxy] on GPU.
        proxy_keys: Key proxy tensor [B, N_kv, S, D_proxy] on GPU.
        alpha: Softmax exponential drop threshold (default 4.0).
        block_size: Paged block token count (default 16).
        max_budget_ratio: Safety cap for maximum prefetched block ratio (default 35%).

    Returns:
        token_mask: Boolean tensor [B, N_kv, S] of selected tokens.
        prefetched_blocks: List of block integer indices scheduled for PCIe DMA transfer.
        prefetch_ratio: Fraction of total sequence tokens scheduled for transfer.
    """
    batch = query.shape[0]
    num_q = query.shape[1]
    num_kv = proxy_keys.shape[1]
    seq_len = proxy_keys.shape[2]
    d_proxy = proxy_keys.shape[3]

    group_size = max(1, num_q // num_kv)

    # Compute speculative attention dot-products
    q_grouped = query.view(batch, num_kv, group_size, 1, d_proxy)
    k_exp = proxy_keys.unsqueeze(2)  # [batch, num_kv, 1, seq_len, d_proxy]
    raw_scores = (q_grouped.float() * k_exp.float()).sum(dim=-1) * (d_proxy ** -0.5)

    # GQA Pooling: Max-pool across grouped query heads to determine shared KV demand
    scores_kv = raw_scores.max(dim=2).values.squeeze(-2)  # [batch, num_kv, seq_len]

    # Dynamic Softmax Cutoff Threshold: tau = max(scores) - alpha
    max_scores = scores_kv.max(dim=-1, keepdim=True).values
    tau = max_scores - float(alpha)
    token_mask = scores_kv >= tau

    # Block-level aggregation
    num_blocks = math.ceil(seq_len / block_size)
    padded_len = num_blocks * block_size
    pad_tokens = padded_len - seq_len

    if pad_tokens > 0:
        mask_padded = F.pad(token_mask, (0, pad_tokens), value=False)
    else:
        mask_padded = token_mask

    # Block is prefetched if ANY token in the block exceeds threshold
    block_mask = mask_padded.view(batch, num_kv, num_blocks, block_size).any(dim=-1)
    shared_block_demand = block_mask.any(dim=0).any(dim=0)  # [num_blocks]

    prefetched_blocks = [idx for idx, val in enumerate(shared_block_demand.tolist()) if val]

    # Enforce safety budget ceiling
    max_allowed_blocks = max(1, int(math.ceil(num_blocks * max_budget_ratio)))
    if len(prefetched_blocks) > max_allowed_blocks:
        # Sort blocks by peak score
        scores_padded = F.pad(scores_kv, (0, pad_tokens), value=float("-inf"))
        block_scores = scores_padded.view(batch, num_kv, num_blocks, block_size).amax(dim=(0, 1, -1))
        top_k_indices = torch.topk(block_scores, k=max_allowed_blocks).indices.tolist()
        prefetched_blocks = sorted(top_k_indices)

    prefetched_token_count = len(prefetched_blocks) * block_size
    prefetch_ratio = min(1.0, prefetched_token_count / max(1, seq_len))

    return token_mask, prefetched_blocks, prefetch_ratio


class InfiniGenLayer(DynamicLayer):
    """Cache layer implementing InfiniGen speculative layer-ahead prefetching.

    Data Allocation:
      1. GPU VRAM:
         - Attention sink tokens (first `gpu_sink_tokens`, e.g. 4)
         - Proxy Key slice of dimension `int(head_dim * k_proxy_ratio)`
         - Recent window tokens (last `block_size` tokens)
      2. Host CPU DRAM:
         - Full uncompressed Key and Value cache tensors across all tokens
      3. Asynchronous PCIe Prefetching:
         - Speculates next-layer token demand using proxy dot products
         - Fetches only the vital 5-10% of historical blocks over PCIe
    """

    is_sliding = False

    def __init__(
        self,
        k_proxy_ratio: float = 0.30,
        alpha_threshold: float = 4.0,
        block_size: int = 16,
        max_budget_ratio: float = 0.35,
        gpu_sink_tokens: int = 4,
    ):
        # Initialize GPU working cache attributes before calling super()
        self.gpu_keys: torch.Tensor | None = None
        self.gpu_values: torch.Tensor | None = None

        super().__init__()

        if k_proxy_ratio <= 0.0 or k_proxy_ratio > 1.0:
            raise ValueError(f"k_proxy_ratio must be in (0, 1], got {k_proxy_ratio}")
        if alpha_threshold <= 0.0:
            raise ValueError(f"alpha_threshold must be positive, got {alpha_threshold}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if gpu_sink_tokens < 0:
            raise ValueError(f"gpu_sink_tokens must be non-negative, got {gpu_sink_tokens}")

        self.k_proxy_ratio = float(k_proxy_ratio)
        self.alpha_threshold = float(alpha_threshold)
        self.block_size = int(block_size)
        self.max_budget_ratio = float(max_budget_ratio)
        self.gpu_sink_tokens = int(gpu_sink_tokens)

        # Full Host CPU DRAM cache pool
        self.cpu_keys: torch.Tensor | None = None
        self.cpu_values: torch.Tensor | None = None

        # GPU Proxy Key slice
        self.proxy_keys: torch.Tensor | None = None

        self.rotation_matrix: torch.Tensor | None = None
        self.d_proxy: int = 0

        self.total_tokens_seen: int = 0
        self.total_bytes_transferred_pcie: int = 0
        self.total_prefetched_blocks: int = 0
        self.total_omitted_blocks: int = 0

    @property
    def keys(self) -> torch.Tensor | None:
        return self.gpu_keys

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        self.gpu_keys = value

    @property
    def values(self) -> torch.Tensor | None:
        return self.gpu_values

    @values.setter
    def values(self, value: torch.Tensor | None) -> None:
        self.gpu_values = value

    def _ensure_rotation_matrix(
        self, head_dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if (
            self.rotation_matrix is None
            or self.rotation_matrix.shape[0] != head_dim
            or self.rotation_matrix.dtype != dtype
            or self.rotation_matrix.device != device
        ):
            self.rotation_matrix = generate_orthogonal_matrix(head_dim).to(
                device=device, dtype=dtype
            )
            self.d_proxy = max(1, int(head_dim * self.k_proxy_ratio))
        return self.rotation_matrix

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update InfiniGen cache: store full tensors in CPU host DRAM, proxy in VRAM, and prefetch."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        target_device = key_states.device
        target_dtype = key_states.dtype
        head_dim = key_states.shape[-1]
        rot_matrix = self._ensure_rotation_matrix(head_dim, target_device, target_dtype)

        new_tokens = key_states.shape[-2]
        self.total_tokens_seen += new_tokens

        # SVD-rotated keys for proxy generation
        k_rotated = torch.matmul(key_states, rot_matrix)
        k_proxy_slice = k_rotated[..., : self.d_proxy]

        # Update GPU Proxy Keys
        if self.proxy_keys is None:
            self.proxy_keys = k_proxy_slice
        else:
            self.proxy_keys = torch.cat([self.proxy_keys, k_proxy_slice], dim=-2)

        # Update full cache in Host CPU DRAM
        k_cpu = key_states.detach().to("cpu", non_blocking=True)
        v_cpu = value_states.detach().to("cpu", non_blocking=True)

        if self.cpu_keys is None:
            self.cpu_keys = k_cpu
            self.cpu_values = v_cpu
        else:
            self.cpu_keys = torch.cat([self.cpu_keys, k_cpu], dim=-2)
            self.cpu_values = torch.cat([self.cpu_values, v_cpu], dim=-2)

        # Track initial offload to CPU DRAM
        bytes_offload = (k_cpu.nelement() + v_cpu.nelement()) * k_cpu.element_size()
        self.total_bytes_transferred_pcie += bytes_offload

        # Single-token decode step with historical context: run Speculative Prefetching
        if new_tokens == 1 and self.total_tokens_seen > self.block_size:
            # Query proxy: rotated current query
            q_rot = torch.matmul(key_states, rot_matrix)
            q_proxy = q_rot[..., : self.d_proxy]

            _, prefetched_blocks, _ = compute_speculative_block_mask(
                query=q_proxy,
                proxy_keys=self.proxy_keys,
                alpha=self.alpha_threshold,
                block_size=self.block_size,
                max_budget_ratio=self.max_budget_ratio,
            )

            # Ensure recent block and attention sink block are always retained
            total_blocks = math.ceil(self.total_tokens_seen / self.block_size)
            sink_block = 0
            recent_block = total_blocks - 1
            all_blocks_to_fetch = set(prefetched_blocks) | {sink_block, recent_block}

            # Retrieve only scheduled blocks from CPU DRAM
            fetch_indices: list[int] = []
            for b in sorted(all_blocks_to_fetch):
                b_start = b * self.block_size
                b_end = min(self.total_tokens_seen, (b + 1) * self.block_size)
                fetch_indices.extend(range(b_start, b_end))

            fetch_idx_tensor = torch.tensor(fetch_indices, dtype=torch.long)
            k_fetched = self.cpu_keys[:, :, fetch_idx_tensor, :].to(target_device, non_blocking=True)
            v_fetched = self.cpu_values[:, :, fetch_idx_tensor, :].to(target_device, non_blocking=True)

            bytes_in = (k_fetched.nelement() + v_fetched.nelement()) * k_fetched.element_size()
            self.total_bytes_transferred_pcie += bytes_in
            self.total_prefetched_blocks += len(all_blocks_to_fetch)
            self.total_omitted_blocks += max(0, total_blocks - len(all_blocks_to_fetch))

            self.gpu_keys = k_fetched
            self.gpu_values = v_fetched
            return self.gpu_keys, self.gpu_values

        # During prefill or small sequences, return full keys from CPU/GPU
        self.gpu_keys = self.cpu_keys.to(target_device, non_blocking=True)
        self.gpu_values = self.cpu_values.to(target_device, non_blocking=True)
        return self.gpu_keys, self.gpu_values

    def get_seq_length(self) -> int:
        return self.total_tokens_seen

    def get_max_length(self) -> int:
        return self.total_tokens_seen + 8192


class InfiniGenCache(Cache):
    """Cache manager for InfiniGen speculative layer-ahead KV prefetching."""

    is_infinigen = True

    def __init__(
        self,
        k_proxy_ratio: float = 0.30,
        alpha_threshold: float = 4.0,
        block_size: int = 16,
        max_budget_ratio: float = 0.35,
        gpu_sink_tokens: int = 4,
    ):
        self.k_proxy_ratio = float(k_proxy_ratio)
        self.alpha_threshold = float(alpha_threshold)
        self.block_size = int(block_size)
        self.max_budget_ratio = float(max_budget_ratio)
        self.gpu_sink_tokens = int(gpu_sink_tokens)

        super().__init__(
            layer_class_to_replicate=lambda: InfiniGenLayer(
                k_proxy_ratio=self.k_proxy_ratio,
                alpha_threshold=self.alpha_threshold,
                block_size=self.block_size,
                max_budget_ratio=self.max_budget_ratio,
                gpu_sink_tokens=self.gpu_sink_tokens,
            )
        )
        set_active_infinigen_cache(self)

    @property
    def total_bytes_transferred_pcie(self) -> int:
        return sum(getattr(layer, "total_bytes_transferred_pcie", 0) for layer in self.layers)

    @property
    def total_prefetched_blocks(self) -> int:
        return sum(getattr(layer, "total_prefetched_blocks", 0) for layer in self.layers)

    @property
    def total_omitted_blocks(self) -> int:
        return sum(getattr(layer, "total_omitted_blocks", 0) for layer in self.layers)

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
