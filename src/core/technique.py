"""The one interface every KV-cache optimization technique implements.

A technique only describes *what varies*: which co-optimization axes it
touches, what engine kwargs it needs, and (optionally) process-level setup
that must run before the engine is built, or a non-default workload. All
the looping, timing, memory measurement, and error handling lives once in
harness.py -- it never needs to be copy-pasted again.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .axis import Axis
from .workload import SyntheticFiller, Workload, WorkloadOutput


class Technique(ABC):
    name: str  # set automatically by the @register("...") decorator
    axes: Axis  # which of STRUCTURAL / SPATIAL / TEMPORAL this technique touches

    def setup(self) -> None:
        """Runs once before the engine is built for each context length.
        Use this for env vars or import-order patches (see techniques/
        lmcache_offload.py for a real example) instead of putting them at
        module level -- module level runs the moment the technique is
        *registered*, which happens for every technique on every run via
        the auto-import in techniques/__init__.py."""
        return None

    @abstractmethod
    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        """Extra kwargs merged into the vLLM LLM(...) constructor."""
        raise NotImplementedError

    def workload(self) -> Workload:
        """Which workload drives this technique. Default: a single cold
        prompt at the target length (fine for pure memory/latency
        techniques). Override for anything involving reuse -- see
        core.workload.ColdWarm."""
        return SyntheticFiller()

    def pytorch_cache_factory(self) -> Any:
        """Optional factory returning a transformers.cache_utils.Cache instance for PyTorch runner.
        Defaults to None (which causes PyTorchRunner to instantiate DynamicCache).
        """
        return None

    def extra_metrics(self, llm, workload_output: WorkloadOutput) -> dict[str, Any]:
        """Optional technique-specific metrics layered on top of the core
        schema, e.g. PCIe bytes transferred or a quality score."""
        return {}
