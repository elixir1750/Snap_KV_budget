"""Training-free PyramidSinkKV utilities for GPTNeoX/Pythia inference.

The implementation intentionally stays small and explicit.  It compresses
GPTNeoX/Pythia-style KV caches after prefill and then uses a manual decoding
loop that keeps RoPE position ids tied to the true logical sequence length,
not to the shortened cache length.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - older transformers fallback
    DynamicCache = None  # type: ignore


LOGGER = logging.getLogger("pyramidsinkkv")


@dataclass
class PyramidSinkKVConfig:
    """Configuration for training-free KV cache compression.

    ``compression_ratio`` is a keep ratio, e.g. 0.5 means "keep about half of
    the prompt KV cache on average".  ``budget_mode`` controls how that average
    ratio is distributed across layers.
    """

    compression_ratio: float = 0.5
    sink_size: int = 4
    recent_size: int = 64
    budget_mode: str = "pyramid"
    score_method: str = "key_norm"
    seed: int = 0
    safe_min_length: int = 8

    def enabled(self) -> bool:
        return self.budget_mode not in ("dense", "no_cache") and self.compression_ratio < 0.999


def normalize_budget_mode(mode: str) -> str:
    if mode == "reversed-pyramid":
        return "reversed"
    return mode


def layer_keep_ratios(
    num_layers: int,
    compression_ratio: float,
    budget_mode: str,
    pyramid_strength: float = 0.8,
) -> List[float]:
    """Return per-layer keep ratios with approximately equal mean.

    For Pythia/GPTNeoX, layer 0 is the lowest layer.  Pyramid mode gives lower
    layers larger budgets and upper layers smaller budgets.  Reversed mode does
    the opposite.  A tiny binary search rescales the profile after clipping so
    all modes keep approximately the requested average ratio.
    """

    if num_layers <= 0:
        return []
    target = float(max(0.0, min(1.0, compression_ratio)))
    mode = normalize_budget_mode(budget_mode)
    if mode in ("dense", "no_cache", "uniform") or num_layers == 1:
        return [1.0 if mode in ("dense", "no_cache") else target for _ in range(num_layers)]
    if mode not in ("pyramid", "reversed", "spindle", "hourglass"):
        raise ValueError(f"Unsupported budget_mode: {budget_mode}")

    if mode in ("pyramid", "reversed"):
        low_to_high = torch.linspace(1.0 + pyramid_strength, 1.0 - pyramid_strength, num_layers)
        profile = low_to_high if mode == "pyramid" else torch.flip(low_to_high, dims=[0])
    else:
        positions = torch.linspace(-1.0, 1.0, num_layers)
        center_weight = 1.0 - positions.abs()
        spindle = 1.0 - pyramid_strength + 2.0 * pyramid_strength * center_weight
        profile = spindle if mode == "spindle" else 2.0 - spindle

    if target <= 0.0:
        return [0.0 for _ in range(num_layers)]
    if target >= 1.0:
        return [1.0 for _ in range(num_layers)]

    lo, hi = 0.0, 1.0
    while torch.clamp(profile * hi, 0.0, 1.0).mean().item() < target and hi < 1000.0:
        hi *= 2.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        mean = torch.clamp(profile * mid, 0.0, 1.0).mean().item()
        if mean < target:
            lo = mid
        else:
            hi = mid
    return torch.clamp(profile * hi, 0.0, 1.0).tolist()


def _cache_to_legacy(cache: Any) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Normalize common HF cache layouts to ``((k, v), ...)``.

    GPTNeoX/Pythia keys and values are expected as ``[batch, heads, seq, dim]``.
    Recent ``transformers`` versions return ``DynamicCache``; older versions
    often return a legacy tuple.  A small amount of compatibility handling keeps
    the project reproducible across course machines.
    """

    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    elif hasattr(cache, "layers"):
        legacy_layers = []
        for layer in cache.layers:
            if not getattr(layer, "is_initialized", False):
                continue
            if not hasattr(layer, "keys") or not hasattr(layer, "values"):
                raise TypeError(f"Unsupported cache layer type: {type(layer)!r}")
            legacy_layers.append((layer.keys, layer.values))
        return tuple(legacy_layers)
    if not isinstance(cache, (tuple, list)):
        raise TypeError(f"Unsupported past_key_values type: {type(cache)!r}")

    # Some prototypes store cache as ``(all_keys, all_values)``.
    if (
        len(cache) == 2
        and isinstance(cache[0], (tuple, list))
        and isinstance(cache[1], (tuple, list))
        and cache[0]
        and cache[1]
        and torch.is_tensor(cache[0][0])
        and torch.is_tensor(cache[1][0])
        and (len(cache[0]) != 2 or len(cache[1]) != 2)
    ):
        keys, values = cache
        return tuple((k, v) for k, v in zip(keys, values))

    legacy = []
    for layer in cache:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise TypeError("Each cache layer must contain at least key and value tensors.")
        legacy.append((layer[0], layer[1]))
    return tuple(legacy)


def _legacy_to_cache(
    legacy: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    original_cache: Any,
) -> Any:
    if hasattr(original_cache, "to_legacy_cache") and DynamicCache is not None:
        return DynamicCache.from_legacy_cache(legacy)
    if hasattr(original_cache, "layers") and DynamicCache is not None:
        return DynamicCache(legacy)
    if isinstance(original_cache, (tuple, list)) and len(original_cache) == 2:
        first = original_cache[0]
        second = original_cache[1]
        if (
            isinstance(first, (tuple, list))
            and isinstance(second, (tuple, list))
            and first
            and second
            and torch.is_tensor(first[0])
            and torch.is_tensor(second[0])
            and (len(first) != 2 or len(second) != 2)
        ):
            return tuple(k for k, _ in legacy), tuple(v for _, v in legacy)
    return legacy


def cache_seq_length(cache: Any, layer_idx: int = 0) -> int:
    legacy = _cache_to_legacy(cache)
    return int(legacy[layer_idx][0].shape[-2])


def approximate_kv_cache_bytes(cache: Any) -> int:
    total = 0
    for key, value in _cache_to_legacy(cache):
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return int(total)


def _mandatory_indices(seq_len: int, sink_size: int, recent_size: int, device: torch.device) -> torch.Tensor:
    pieces = []
    if sink_size > 0:
        pieces.append(torch.arange(0, min(sink_size, seq_len), device=device, dtype=torch.long))
    if recent_size > 0:
        pieces.append(torch.arange(max(0, seq_len - recent_size), seq_len, device=device, dtype=torch.long))
    if not pieces:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.unique(torch.cat(pieces), sorted=True)


def _middle_candidates(seq_len: int, sink_size: int, recent_size: int, device: torch.device) -> torch.Tensor:
    start = min(max(sink_size, 0), seq_len)
    end = max(start, seq_len - max(recent_size, 0))
    return torch.arange(start, end, device=device, dtype=torch.long)


def _key_norm_scores(key: torch.Tensor) -> torch.Tensor:
    # Expected key shape for GPTNeoX/Pythia cache: [batch, heads, seq_len, head_dim].
    if key.ndim != 4:
        raise ValueError(f"Expected key tensor [batch, heads, seq, dim], got shape {tuple(key.shape)}")
    return key.float().norm(dim=-1).mean(dim=(0, 1))


def _attention_scores(attention: Optional[torch.Tensor], seq_len: int, recent_size: int) -> Optional[torch.Tensor]:
    if attention is None or not torch.is_tensor(attention):
        return None
    if attention.ndim != 4 or attention.shape[-1] != seq_len:
        return None
    window = min(max(recent_size, 1), attention.shape[-2])
    return attention[:, :, -window:, :].float().mean(dim=(0, 1, 2))


def select_token_indices(
    key: torch.Tensor,
    keep_count: int,
    config: PyramidSinkKVConfig,
    layer_idx: int,
    attention: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, str]:
    """Select ordered token indices for one layer.

    Sink and recent regions are always retained first.  Middle tokens are then
    selected by random, key-norm, or attention score.  The returned indices are
    sorted so the compressed cache preserves original token order.
    """

    seq_len = int(key.shape[-2])
    device = key.device
    keep_count = min(seq_len, max(1, int(keep_count)))
    mandatory = _mandatory_indices(seq_len, config.sink_size, config.recent_size, device)
    candidates = _middle_candidates(seq_len, config.sink_size, config.recent_size, device)
    keep_count = max(keep_count, int(mandatory.numel()))
    remaining = keep_count - int(mandatory.numel())

    if remaining <= 0 or candidates.numel() == 0:
        selected = mandatory
        selected = torch.unique(selected, sorted=True)
        assert_valid_selected_indices(selected, seq_len)
        return selected, config.score_method
    if remaining >= candidates.numel():
        selected = torch.cat([mandatory, candidates])
        selected = torch.unique(selected, sorted=True)
        assert_valid_selected_indices(selected, seq_len)
        return selected, config.score_method

    method = config.score_method
    if method == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.seed) + int(layer_idx))
        perm = torch.randperm(int(candidates.numel()), generator=generator)[:remaining].to(device)
        middle = candidates[perm]
    else:
        if method == "attention":
            scores = _attention_scores(attention, seq_len, config.recent_size)
            if scores is None:
                LOGGER.warning(
                    "Attention weights are unavailable for layer %s; falling back to key_norm scoring.",
                    layer_idx,
                )
                method = "key_norm"
        if method == "key_norm":
            scores = _key_norm_scores(key)
        middle_scores = scores[candidates]
        top = torch.topk(middle_scores, k=remaining, largest=True).indices
        middle = candidates[top]

    selected = torch.cat([mandatory, middle])
    selected = torch.unique(selected, sorted=True)
    assert_valid_selected_indices(selected, seq_len)
    return selected, method


def assert_valid_selected_indices(indices: torch.Tensor, seq_len: int) -> None:
    """Defensive checks for compressed KV gather indices."""

    if indices.ndim != 1:
        raise AssertionError(f"selected indices must be 1D, got {tuple(indices.shape)}")
    if indices.numel() == 0:
        raise AssertionError("selected indices must not be empty")
    if torch.any(indices < 0) or torch.any(indices >= seq_len):
        raise AssertionError(f"selected indices out of range [0, {seq_len})")
    if indices.numel() > 1:
        diffs = indices[1:] - indices[:-1]
        if torch.any(diffs <= 0):
            raise AssertionError("selected indices must be sorted and unique")


def compress_past_key_values(
    past_key_values: Any,
    config: PyramidSinkKVConfig,
    attentions: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Compress a GPTNeoX/Pythia KV cache layer by layer after prefill."""

    legacy = _cache_to_legacy(past_key_values)
    num_layers = len(legacy)
    ratios = layer_keep_ratios(num_layers, config.compression_ratio, config.budget_mode)
    compressed_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
    per_layer = []

    for layer_idx, ((key, value), ratio) in enumerate(zip(legacy, ratios)):
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "PyramidSinkKV expects GPTNeoX cache tensors shaped "
                f"[batch, heads, seq, head_dim], got {tuple(key.shape)} and {tuple(value.shape)}"
            )
        seq_len = int(key.shape[-2])
        safe_min = min(seq_len, max(1, int(config.safe_min_length)))
        target_keep = int(round(seq_len * float(ratio)))
        target_keep = max(target_keep, safe_min)
        target_keep = min(seq_len, target_keep)
        attention = attentions[layer_idx] if attentions is not None and layer_idx < len(attentions) else None
        indices, used_method = select_token_indices(key, target_keep, config, layer_idx, attention)

        gather_idx = indices.view(1, 1, -1, 1).expand(key.shape[0], key.shape[1], -1, key.shape[-1])
        compressed_key = torch.gather(key, dim=2, index=gather_idx)
        compressed_value = torch.gather(value, dim=2, index=gather_idx)
        compressed_layers.append((compressed_key.contiguous(), compressed_value.contiguous()))
        per_layer.append(
            {
                "layer": layer_idx,
                "original_seq_len": seq_len,
                "compressed_seq_len": int(indices.numel()),
                "target_keep_ratio": float(ratio),
                "actual_keep_ratio": float(indices.numel() / max(seq_len, 1)),
                "score_method": used_method,
            }
        )

    original_tokens = sum(item["original_seq_len"] for item in per_layer)
    compressed_tokens = sum(item["compressed_seq_len"] for item in per_layer)
    compressed_cache = _legacy_to_cache(tuple(compressed_layers), past_key_values)
    stats = {
        "config": asdict(config),
        "per_layer": per_layer,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "achieved_compression_ratio": compressed_tokens / max(original_tokens, 1),
    }
    return compressed_cache, stats


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    config: Optional[PyramidSinkKVConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Greedy generation with optional post-prefill PyramidSinkKV compression."""

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    config = config or PyramidSinkKVConfig(budget_mode="dense", compression_ratio=1.0)
    dense = not config.enabled()

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    logical_seq_len = int(input_ids.shape[1])
    output_attentions = (not dense) and config.score_method == "attention"

    synchronize_if_cuda(device)
    start = time.perf_counter()
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids,
            use_cache=True,
            output_attentions=output_attentions,
        )
        past_key_values = prefill_outputs.past_key_values
        compression_stats: Dict[str, Any] = {
            "achieved_compression_ratio": 1.0,
            "original_tokens": logical_seq_len * len(_cache_to_legacy(past_key_values)),
            "compressed_tokens": logical_seq_len * len(_cache_to_legacy(past_key_values)),
            "per_layer": [],
        }
        if not dense:
            past_key_values, compression_stats = compress_past_key_values(
                past_key_values,
                config,
                prefill_outputs.attentions,
            )
            compressed_len = cache_seq_length(past_key_values)
            # RoPE sanity check: after compression, the cache length is allowed
            # to be shorter than the true logical sequence.  Decoding below
            # therefore passes explicit absolute position_ids/cache_position.
            assert compressed_len <= logical_seq_len
            if compressed_len != logical_seq_len:
                compression_stats["rope_sanity"] = {
                    "logical_seq_len": logical_seq_len,
                    "compressed_cache_len": compressed_len,
                }

        next_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1)
    synchronize_if_cuda(device)
    ttft = time.perf_counter() - start

    generated: List[int] = [int(next_token.item())] if max_new_tokens > 0 else []
    decode_times: List[float] = []

    for _ in range(max(0, max_new_tokens - 1)):
        current_ids = next_token.view(1, 1).to(device)
        # GPTNeoX uses RoPE.  Because KV compression deletes cache entries,
        # ``past_key_values.get_seq_length()`` is no longer the absolute token
        # position.  Always pass the true logical position for the new token.
        position_ids = torch.full((1, 1), logical_seq_len, dtype=torch.long, device=device)
        cache_position = torch.arange(logical_seq_len, logical_seq_len + 1, dtype=torch.long, device=device)
        synchronize_if_cuda(device)
        step_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        synchronize_if_cuda(device)
        decode_times.append(time.perf_counter() - step_start)
        logical_seq_len += 1
        generated.append(int(next_token.item()))

    total_time = ttft + sum(decode_times)
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    generated_tensor = torch.tensor(generated, dtype=input_ids.dtype)
    full_ids = torch.cat([input_ids[0].detach().cpu(), generated_tensor]) if generated else input_ids[0].detach().cpu()
    full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    tpot = sum(decode_times) / len(decode_times) if decode_times else 0.0
    throughput = len(generated) / total_time if total_time > 0 else math.inf

    return {
        "method_config": asdict(config),
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": len(generated),
        "ttft": ttft,
        "tpot": tpot,
        "throughput": throughput,
        "total_time": total_time,
        "kv_cache_memory_bytes": approximate_kv_cache_bytes(past_key_values),
        "kv_cache_memory_mb": approximate_kv_cache_bytes(past_key_values) / (1024**2),
        "compression": compression_stats,
        "generated_text": generated_text,
        "full_text": full_text,
    }


def generate_no_cache(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Greedy generation baseline with ``use_cache=False``.

    This is intentionally slow: every decoding step re-runs the full current
    sequence and stores no KV cache.  It is useful as an educational baseline,
    but the dense baseline with full KV cache is the standard inference setup.
    """

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    encoded = tokenizer(prompt, return_tensors="pt")
    full_ids = encoded.input_ids.to(device)
    generated: List[int] = []
    decode_times: List[float] = []

    synchronize_if_cuda(device)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=full_ids, use_cache=False)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    synchronize_if_cuda(device)
    ttft = time.perf_counter() - start

    if max_new_tokens > 0:
        generated.append(int(next_token.item()))
        full_ids = torch.cat([full_ids, next_token.view(1, 1).to(device)], dim=1)

    for _ in range(max(0, max_new_tokens - 1)):
        synchronize_if_cuda(device)
        step_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids=full_ids, use_cache=False)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        synchronize_if_cuda(device)
        decode_times.append(time.perf_counter() - step_start)
        generated.append(int(next_token.item()))
        full_ids = torch.cat([full_ids, next_token.view(1, 1).to(device)], dim=1)

    total_time = ttft + sum(decode_times)
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    full_text = tokenizer.decode(full_ids[0].detach().cpu(), skip_special_tokens=True)
    tpot = sum(decode_times) / len(decode_times) if decode_times else 0.0
    throughput = len(generated) / total_time if total_time > 0 else math.inf

    return {
        "method_config": asdict(PyramidSinkKVConfig(compression_ratio=1.0, budget_mode="no_cache")),
        "prompt_tokens": int(encoded.input_ids.shape[1]),
        "generated_tokens": len(generated),
        "ttft": ttft,
        "tpot": tpot,
        "throughput": throughput,
        "total_time": total_time,
        "kv_cache_memory_bytes": 0,
        "kv_cache_memory_mb": 0.0,
        "compression": {
            "achieved_compression_ratio": None,
            "original_tokens": 0,
            "compressed_tokens": 0,
            "per_layer": [],
            "note": "use_cache=False; no KV cache is stored",
        },
        "generated_text": generated_text,
        "full_text": full_text,
    }


def parse_torch_dtype(dtype: str, device: str = "cpu") -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "auto":
        return torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str = "auto",
    dtype: str = "auto",
    attn_implementation: Optional[str] = None,
) -> Tuple[Any, Any, torch.device]:
    """Load Pythia/GPTNeoX while respecting HuggingFace environment variables."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    torch_dtype = parse_torch_dtype(dtype, resolved_device.type)
    kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.to(resolved_device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, resolved_device
