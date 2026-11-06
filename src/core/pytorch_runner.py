"""Generic PyTorch inference runner supporting Hugging Face Transformers and Cache abstractions.

Serves as the scientific control execution engine for evaluating KV-cache co-optimization
techniques without continuous batching or CUDA kernel compiler bias.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicCache


@dataclass
class PyTorchCompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float = 0.0
    logprobs: list[dict[int, float]] | None = None
    finish_reason: str = "length"


@dataclass
class PyTorchRequestMetrics:
    first_token_latency: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0
    scheduled_ts: float = 0.0


@dataclass
class PyTorchRequestOutput:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    outputs: list[PyTorchCompletionOutput]
    finished: bool = True
    metrics: PyTorchRequestMetrics | None = None
    diagnostics: Any = None


class PyTorchRunner:
    """Generic inference runner using Hugging Face AutoModelForCausalLM and Cache abstractions."""

    def __init__(
        self,
        model_name_or_path: str,
        revision: str | None = None,
        device: str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
        dtype: torch.dtype | None = None,
        cache_factory: Callable[[], Cache] | None = None,
        trust_remote_code: bool = True,
        dry_run: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.device = device
        self.cache_factory = cache_factory or (lambda: DynamicCache())
        self.dry_run = dry_run
        self.model: Any = None
        self.tokenizer: Any = None

        if dtype is None:
            if torch is not None and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16 if (torch is not None and torch.cuda.is_available()) else torch.float32
        else:
            self.dtype = dtype

        if not self.dry_run:
            self._load_model(trust_remote_code)

    def _load_model(self, trust_remote_code: bool) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[PyTorchRunner] Loading tokenizer from '{self.model_name_or_path}' (revision: {self.revision})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"[PyTorchRunner] Loading model from '{self.model_name_or_path}' on {self.device} (dtype: {self.dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            torch_dtype=self.dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
        )
        self.model.eval()

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def generate(
        self,
        prompts: list[str] | str,
        sampling_params: Any = None,
        max_tokens: int = 64,
        logprobs_topk: int = 5,
        chunk_size: int = 1024,
        post_hoc_hook: Any = None,
    ) -> list[PyTorchRequestOutput]:
        """Execute autoregressive generation across requested prompts."""
        if isinstance(prompts, str):
            prompts = [prompts]

        if sampling_params is not None:
            max_tokens = getattr(sampling_params, "max_tokens", max_tokens)
            logprobs_topk = getattr(sampling_params, "logprobs", logprobs_topk) or logprobs_topk

        if self.dry_run:
            results: list[PyTorchRequestOutput] = []
            for i, p in enumerate(prompts):
                fake_prompt_ids = list(range(100))
                fake_out_ids = list(range(max_tokens))
                metrics = PyTorchRequestMetrics(
                    first_token_latency=0.01,
                    first_token_ts=1.0,
                    last_token_ts=1.0 + max_tokens * 0.03,
                    scheduled_ts=0.0,
                )
                comp = PyTorchCompletionOutput(
                    index=0,
                    text="[DRY RUN GENERATION]",
                    token_ids=fake_out_ids,
                    logprobs=[{tid: -0.5} for tid in fake_out_ids],
                )
                results.append(
                    PyTorchRequestOutput(
                        request_id=f"dry_run_{i}",
                        prompt=p,
                        prompt_token_ids=fake_prompt_ids,
                        outputs=[comp],
                        metrics=metrics,
                    )
                )
            return results

        request_outputs: list[PyTorchRequestOutput] = []

        for req_idx, prompt in enumerate(prompts):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            prompt_token_ids = input_ids[0].tolist()
            seq_len = input_ids.shape[1]

            cache = self.cache_factory()

            # Record prefill latency via CUDA events
            use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
            start_wall = time.perf_counter()

            if use_cuda:
                start_prefill = torch.cuda.Event(enable_timing=True)
                end_prefill = torch.cuda.Event(enable_timing=True)
                start_prefill.record()

            # Execute Prefill (Chunked for memory safety when prompt > chunk_size)
            if seq_len > chunk_size:
                num_chunks = (seq_len + chunk_size - 1) // chunk_size
                for chunk_idx in range(num_chunks):
                    c_start = chunk_idx * chunk_size
                    c_end = min(seq_len, (chunk_idx + 1) * chunk_size)
                    chunk_input = input_ids[:, c_start:c_end]
                    chunk_pos = torch.arange(c_start, c_end, dtype=torch.long, device=self.device).unsqueeze(0)

                    with torch.no_grad():
                        out = self.model(
                            chunk_input,
                            past_key_values=cache,
                            position_ids=chunk_pos,
                            use_cache=True,
                        )
                    # Prune intermediate chunks if cache supports chunk pruning
                    if hasattr(cache, "prune_after_chunk"):
                        cache.prune_after_chunk()
                last_logits = out.logits[:, -1, :]
            else:
                pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    out = self.model(
                        input_ids,
                        past_key_values=cache,
                        position_ids=pos_ids,
                        use_cache=True,
                    )
                if hasattr(cache, "prune_after_chunk"):
                    cache.prune_after_chunk()
                last_logits = out.logits[:, -1, :]

            if use_cuda:
                end_prefill.record()
                torch.cuda.synchronize()
                prefill_sec = max(0.0001, start_prefill.elapsed_time(end_prefill) / 1000.0)
            else:
                prefill_sec = max(0.0001, time.perf_counter() - start_wall)

            # Prefill first generated token selection
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            generated_ids: list[int] = [int(next_token[0, 0].item())]
            step_logprobs: list[dict[int, float]] = []

            # Extract step 0 logprobs
            lps = F.log_softmax(last_logits, dim=-1)
            top_lp, top_idx = torch.topk(lps, k=min(logprobs_topk, lps.shape[-1]), dim=-1)
            step_dict: dict[int, float] = {}
            for k_idx in range(top_lp.shape[-1]):
                step_dict[int(top_idx[0, k_idx].item())] = float(top_lp[0, k_idx].item())
            step_logprobs.append(step_dict)

            # Autoregressive Decode Loop
            step_events: list[tuple[Any, Any]] = []
            curr_pos = seq_len

            for _ in range(max_tokens - 1):
                if use_cuda:
                    s_ev = torch.cuda.Event(enable_timing=True)
                    e_ev = torch.cuda.Event(enable_timing=True)
                    s_ev.record()

                pos_tensor = torch.tensor([[curr_pos]], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    step_out = self.model(
                        next_token,
                        past_key_values=cache,
                        position_ids=pos_tensor,
                        use_cache=True,
                    )

                if use_cuda:
                    e_ev.record()
                    step_events.append((s_ev, e_ev))

                logits_step = step_out.logits[:, -1, :]
                next_token = torch.argmax(logits_step, dim=-1, keepdim=True)
                generated_ids.append(int(next_token[0, 0].item()))

                lps_step = F.log_softmax(logits_step, dim=-1)
                top_lp_s, top_idx_s = torch.topk(lps_step, k=min(logprobs_topk, lps_step.shape[-1]), dim=-1)
                step_d: dict[int, float] = {}
                for k_i in range(top_lp_s.shape[-1]):
                    step_d[int(top_idx_s[0, k_i].item())] = float(top_lp_s[0, k_i].item())
                step_logprobs.append(step_d)

                curr_pos += 1
                if int(next_token[0, 0].item()) == self.tokenizer.eos_token_id:
                    break

            # Synchronize once after decode loop completes
            decode_sec = 0.0
            if use_cuda and step_events:
                torch.cuda.synchronize()
                total_step_ms = sum(s.elapsed_time(e) for s, e in step_events)
                decode_sec = total_step_ms / 1000.0
            elif not use_cuda:
                decode_sec = max(0.0, time.perf_counter() - start_wall - prefill_sec)

            first_ts = start_wall + prefill_sec
            last_ts = first_ts + decode_sec

            metrics = PyTorchRequestMetrics(
                first_token_latency=round(prefill_sec, 4),
                first_token_ts=first_ts,
                last_token_ts=last_ts,
                scheduled_ts=start_wall,
            )

            gen_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            comp = PyTorchCompletionOutput(
                index=0,
                text=gen_text,
                token_ids=generated_ids,
                logprobs=step_logprobs,
            )

            req_out = PyTorchRequestOutput(
                request_id=f"pt_req_{req_idx}",
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                outputs=[comp],
                metrics=metrics,
            )
            if post_hoc_hook is not None:
                try:
                    req_out.diagnostics = post_hoc_hook(cache, req_idx, req_out)
                except Exception as hook_err:
                    req_out.diagnostics = {"error": str(hook_err)}
            request_outputs.append(req_out)

            # Explicit cleanup of per-request GPU tensors to prevent allocator fragmentation
            del input_ids, cache, out, next_token, last_logits
            if "step_out" in locals():
                del step_out
            if "pos_tensor" in locals():
                del pos_tensor
            if use_cuda:
                torch.cuda.empty_cache()

        return request_outputs
