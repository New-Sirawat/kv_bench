from __future__ import annotations

from collections.abc import Iterator
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer
from src.techniques.streamingllm import evict_streaming_kv_cache


class StreamingLayer(DynamicLayer):
    """Cache layer implementing dual-region StreamingLLM eviction (sink + recent window)."""

    is_sliding = False

    def __init__(self, sink_tokens: int = 4, window_size: int = 2048):
        super().__init__()
        if int(sink_tokens) < 0:
            raise ValueError(f"sink_tokens must be >= 0, got {sink_tokens}")
        if int(window_size) <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")

        self.sink_tokens = int(sink_tokens)
        self.window_size = int(window_size)
        self.original_positions: list[int] = []
        self.total_tokens_seen: int = 0

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update key and value cache tensors and evict middle tokens in decode phase."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

        new_tokens = key_states.shape[-2]
        self.original_positions.extend(range(self.total_tokens_seen, self.total_tokens_seen + new_tokens))
        self.total_tokens_seen += new_tokens

        # In autoregressive decode phase (seq_len == 1), evict immediately to clamp memory
        if key_states.shape[-2] == 1:
            self.keys, self.values = evict_streaming_kv_cache(
                self.keys,
                self.values,
                sink_tokens=self.sink_tokens,
                window_size=self.window_size,
                seq_dim=-2,
            )
            if len(self.original_positions) > self.sink_tokens + self.window_size:
                self.original_positions = (
                    self.original_positions[: self.sink_tokens]
                    + self.original_positions[-self.window_size :]
                )

        return self.keys, self.values

    def prune(self) -> None:
        """Explicitly evict middle tokens after chunked or full prefill."""
        if self.is_initialized and self.keys is not None and self.values is not None:
            self.keys, self.values = evict_streaming_kv_cache(
                self.keys,
                self.values,
                sink_tokens=self.sink_tokens,
                window_size=self.window_size,
                seq_dim=-2,
            )
            if len(self.original_positions) > self.sink_tokens + self.window_size:
                self.original_positions = (
                    self.original_positions[: self.sink_tokens]
                    + self.original_positions[-self.window_size :]
                )

    def get_max_length(self) -> int:
        return self.sink_tokens + self.window_size


class StreamingCache(Cache):
    """Subclass of transformers.cache_utils.Cache implementing exact StreamingLLM dual-region retention.

    Anchors the first N_sink attention sink tokens while maintaining a rolling local
    sliding window of W recent tokens, evicting middle tokens to bound active KV-cache
    memory to O(1) across arbitrary sequence lengths.
    """

    def __init__(
        self,
        sink_tokens: int = 4,
        window_size: int = 2048,
    ):
        if int(sink_tokens) < 0:
            raise ValueError(f"sink_tokens must be >= 0, got {sink_tokens}")
        if int(window_size) <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")

        self.sink_tokens = int(sink_tokens)
        self.window_size = int(window_size)
        self.original_positions: list[int] = []
        self.total_tokens_seen: int = 0

        super().__init__(
            layer_class_to_replicate=lambda: StreamingLayer(
                sink_tokens=self.sink_tokens,
                window_size=self.window_size,
            )
        )

    def prune_after_chunk(self) -> None:
        """Evict middle tokens across all initialized layers after prefill chunk processing."""
        for layer in self.layers:
            if hasattr(layer, "prune"):
                layer.prune()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self) -> int:
        return self.sink_tokens + self.window_size

    def __iter__(self) -> Iterator[tuple[Any, Any, Any]]:
        for layer in self.layers:
            yield getattr(layer, "keys", None), getattr(layer, "values", None), None
