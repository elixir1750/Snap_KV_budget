"""Training-free PyramidSinkKV utilities for GPTNeoX/Pythia inference.

The implementation intentionally stays small and explicit.  It compresses
GPTNeoX/Pythia-style KV caches after prefill and then uses a manual decoding
loop that keeps RoPE position ids tied to the true logical sequence length,
not to the shortened cache length.
"""

from __future__ import annotations

import logging
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

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
    observation_window: int = 32
    snapkv_pooling_kernel: int = 1
    snapkv_head_aggregation: str = "mean"
    seed: int = 0
    safe_min_length: int = 8
    debug_selection: bool = False
    debug_dir: str = "results/debug_selection"

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


def cache_seq_lengths(cache: Any) -> List[int]:
    return [int(key.shape[-2]) for key, _ in _cache_to_legacy(cache)]


def get_seq_len_from_kv(key: torch.Tensor) -> int:
    """Return the KV sequence length for GPTNeoX/Pythia cache tensors."""

    if key.ndim != 4:
        raise ValueError(f"Expected key tensor [batch, heads, seq, dim], got shape {tuple(key.shape)}")
    return int(key.shape[-2])


def approximate_kv_cache_bytes(cache: Any) -> int:
    total = 0
    for key, value in _cache_to_legacy(cache):
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return int(total)


def decode_attention_mask_for_cache(cache: Any, device: torch.device) -> Optional[torch.Tensor]:
    """Return a decode mask only when one global mask fits every layer.

    Layer-wise budgets can produce different physical cache lengths per layer.
    HuggingFace GPTNeoX accepts only one attention_mask, so a global mask is
    valid only when all layers have the same compressed length.  With batch size
    one and no padding, omitting the mask lets each layer attend over its own
    physical K/V length while explicit position_ids/cache_position preserve
    RoPE's logical absolute positions.
    """

    lengths = cache_seq_lengths(cache)
    if not lengths or len(set(lengths)) != 1:
        return None
    return torch.ones((1, lengths[0] + 1), dtype=torch.long, device=device)


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
    return _snapkv_attention_scores(attention, seq_len, recent_size)


def _pool_scores_1d(scores: torch.Tensor, pooling_kernel: int) -> torch.Tensor:
    """Apply length-preserving 1D average pooling over the last dimension."""

    kernel = int(pooling_kernel)
    seq_len = int(scores.shape[-1])
    if kernel <= 1 or seq_len <= 1:
        return scores
    kernel = min(kernel, seq_len)
    pad_left = (kernel - 1) // 2
    pad_right = kernel // 2
    leading_shape = scores.shape[:-1]
    flat = scores.reshape(-1, 1, seq_len)
    padded = F.pad(flat, (pad_left, pad_right), mode="replicate")
    pooled = F.avg_pool1d(padded, kernel_size=kernel, stride=1)
    return pooled.reshape(*leading_shape, seq_len)


def _snapkv_attention_scores(
    attention: Optional[torch.Tensor],
    seq_len: int,
    observation_window: int,
    pooling_kernel: int = 1,
    head_aggregation: str = "mean",
) -> Optional[torch.Tensor]:
    """Compute SnapKV-style scores from recent queries to all prompt keys.

    HF GPTNeoX attentions are normally ``[batch, heads, query_len, key_len]``.
    We explicitly validate this shape and average the last observation-window
    query rows over batch and query positions.  Head scores can then be
    aggregated with ``mean`` or ``max`` to yield ``[seq_len]``, or kept as
    ``[heads, seq_len]`` with ``per_head`` for true per-head SnapKV selection.
    When ``pooling_kernel`` is greater than one, apply a length-preserving 1D
    average pooling pass over the sequence dimension.  This keeps the current
    head-agnostic cache format while adding SnapKV's local continuity bias.
    """

    if attention is None or not torch.is_tensor(attention):
        return None
    if attention.ndim != 4:
        return None
    if int(attention.shape[-1]) < seq_len:
        return None
    window = min(max(int(observation_window), 1), int(attention.shape[-2]))
    key_start = int(attention.shape[-1]) - seq_len
    recent_attn = attention[:, :, -window:, key_start:]
    per_head_score = recent_attn.float().mean(dim=(0, 2))
    if per_head_score.ndim != 2 or int(per_head_score.shape[-1]) != seq_len:
        return None
    aggregation = str(head_aggregation).lower()
    if aggregation == "per_head":
        score = _pool_scores_1d(per_head_score, pooling_kernel)
        if score.ndim != 2 or int(score.shape[-1]) != seq_len:
            return None
        return score
    if aggregation == "mean":
        score = per_head_score.mean(dim=0)
    elif aggregation == "max":
        score = per_head_score.max(dim=0).values
    else:
        raise ValueError(f"Unsupported snapkv_head_aggregation: {head_aggregation}")
    if score.ndim != 1 or int(score.shape[0]) != seq_len:
        return None
    score = _pool_scores_1d(score, pooling_kernel)
    if score.ndim != 1 or int(score.shape[0]) != seq_len:
        return None
    return score


def _select_middle_by_method(
    key: torch.Tensor,
    candidates: torch.Tensor,
    remaining: int,
    method: str,
    config: PyramidSinkKVConfig,
    layer_idx: int,
    attention: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, str, bool]:
    """Select middle-token indices and report the method actually used."""

    seq_len = get_seq_len_from_kv(key)
    fallback = False
    if method == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.seed) + int(layer_idx))
        perm = torch.randperm(int(candidates.numel()), generator=generator)[:remaining].to(key.device)
        return candidates[perm], method, fallback

    scores: Optional[torch.Tensor] = None
    if method == "attention":
        scores = _attention_scores(attention, seq_len, config.recent_size)
        if scores is None:
            LOGGER.warning(
                "Attention weights are unavailable for layer %s; falling back to key_norm scoring.",
                layer_idx,
            )
            method = "key_norm"
            fallback = True
    elif method == "snapkv":
        scores = _snapkv_attention_scores(
            attention,
            seq_len,
            config.observation_window,
            config.snapkv_pooling_kernel,
            config.snapkv_head_aggregation,
        )
        if scores is None:
            LOGGER.warning(
                "SnapKV attention weights are unavailable; falling back to key_norm. layer=%s",
                layer_idx,
            )
            method = "key_norm"
            fallback = True

    if method == "key_norm":
        scores = _key_norm_scores(key)
    if scores is None:
        raise ValueError(f"Unsupported score_method: {method}")
    if scores.ndim == 1:
        middle_scores = scores[candidates]
        top = torch.topk(middle_scores, k=remaining, largest=True).indices
        return candidates[top], method, fallback
    if scores.ndim == 2:
        middle_scores = scores[:, candidates]
        top = torch.topk(middle_scores, k=remaining, dim=-1, largest=True).indices
        return candidates[top], method, fallback
    raise ValueError(f"Expected 1D or 2D scores, got shape {tuple(scores.shape)}")


def gather_kv_by_indices(key: torch.Tensor, value: torch.Tensor, selected_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V along the sequence dimension only.

    ``selected_indices`` may be 1D ``[keep_len]`` for head-agnostic selection
    or 2D ``[heads, keep_len]`` for per-head SnapKV selection.
    """

    seq_len = get_seq_len_from_kv(key)
    assert_valid_selected_indices(selected_indices, seq_len)
    if value.ndim != 4 or int(value.shape[-2]) != seq_len:
        raise ValueError(f"Expected value tensor [batch, heads, seq, dim], got shape {tuple(value.shape)}")
    if selected_indices.ndim == 1:
        gather_idx = selected_indices.view(1, 1, -1, 1).expand(key.shape[0], key.shape[1], -1, key.shape[-1])
        value_idx = selected_indices.view(1, 1, -1, 1).expand(value.shape[0], value.shape[1], -1, value.shape[-1])
    else:
        if int(selected_indices.shape[0]) != int(key.shape[1]):
            raise AssertionError(
                f"per-head selected indices must have one row per head: "
                f"indices={tuple(selected_indices.shape)} key={tuple(key.shape)}"
            )
        gather_idx = selected_indices.view(1, key.shape[1], -1, 1).expand(key.shape[0], key.shape[1], -1, key.shape[-1])
        value_idx = selected_indices.view(1, value.shape[1], -1, 1).expand(value.shape[0], value.shape[1], -1, value.shape[-1])
    compressed_key = torch.gather(key, dim=2, index=gather_idx)
    compressed_value = torch.gather(value, dim=2, index=value_idx)
    expected = int(selected_indices.shape[-1])
    if int(compressed_key.shape[-2]) != expected or int(compressed_value.shape[-2]) != expected:
        raise AssertionError("compressed K/V length does not match selected indices length")
    return compressed_key.contiguous(), compressed_value.contiguous()


def _selection_summary(indices: torch.Tensor, seq_len: int) -> Dict[str, Any]:
    cpu = indices.detach().cpu()
    if cpu.ndim == 1:
        rows = cpu.view(1, -1)
    else:
        rows = cpu
    gaps = (rows[:, 1:] - rows[:, :-1]) if rows.shape[-1] > 1 else torch.empty(0, dtype=cpu.dtype)
    payload = {
        "selected_indices": [int(x) for x in rows[0].tolist()],
        "num_selected": int(rows.shape[-1]),
        "num_heads": int(rows.shape[0]) if cpu.ndim == 2 else None,
        "min_selected_index": int(rows.min().item()) if rows.numel() else None,
        "max_selected_index": int(rows.max().item()) if rows.numel() else None,
        "average_selected_index": float(rows.float().mean().item()) if rows.numel() else None,
        "max_gap": int(gaps.max().item()) if gaps.numel() else 0,
        "average_gap": float(gaps.float().mean().item()) if gaps.numel() else 0.0,
        "seq_len": int(seq_len),
    }
    if cpu.ndim == 2:
        payload["selected_indices_by_head"] = [[int(x) for x in row.tolist()] for row in rows]
    return payload


def _overlap_ratio(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()
    left_rows = left_cpu.view(1, -1) if left_cpu.ndim == 1 else left_cpu
    right_rows = right_cpu.view(1, -1) if right_cpu.ndim == 1 else right_cpu
    overlaps = []
    for idx, left_row in enumerate(left_rows):
        right_row = right_rows[idx] if right_rows.shape[0] == left_rows.shape[0] else right_rows[0]
        left_set = set(int(x) for x in left_row.tolist())
        right_set = set(int(x) for x in right_row.tolist())
        overlaps.append(len(left_set & right_set) / max(len(left_set), 1))
    return float(sum(overlaps) / max(len(overlaps), 1))


def _save_selection_debug(
    key: torch.Tensor,
    selected: torch.Tensor,
    keep_count: int,
    config: PyramidSinkKVConfig,
    layer_idx: int,
    attention: Optional[torch.Tensor],
    used_method: str,
) -> None:
    if not config.debug_selection:
        return
    seq_len = get_seq_len_from_kv(key)
    mandatory = _mandatory_indices(seq_len, config.sink_size, config.recent_size, key.device)
    candidates = _middle_candidates(seq_len, config.sink_size, config.recent_size, key.device)
    remaining = max(0, min(int(keep_count), seq_len) - int(mandatory.numel()))
    payload = _selection_summary(selected, seq_len)
    payload.update({"layer": int(layer_idx), "score_method": config.score_method, "used_score_method": used_method})
    if remaining > 0 and candidates.numel() > 0:
        baseline_config = PyramidSinkKVConfig(**{**asdict(config), "debug_selection": False})
        if used_method != "random":
            random_middle, _, _ = _select_middle_by_method(
                key, candidates, min(remaining, int(candidates.numel())), "random", baseline_config, layer_idx, attention
            )
            random_selected = torch.unique(torch.cat([mandatory, random_middle]), sorted=True)
            payload["overlap_with_random"] = _overlap_ratio(selected, random_selected)
        if used_method != "key_norm":
            key_norm_middle, _, _ = _select_middle_by_method(
                key, candidates, min(remaining, int(candidates.numel())), "key_norm", baseline_config, layer_idx, attention
            )
            key_norm_selected = torch.unique(torch.cat([mandatory, key_norm_middle]), sorted=True)
            payload["overlap_with_key_norm"] = _overlap_ratio(selected, key_norm_selected)
    debug_dir = Path(config.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    out = debug_dir / f"{config.budget_mode}_{config.score_method}_layer{layer_idx:02d}_{os.getpid()}_{timestamp}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def select_token_indices(
    key: torch.Tensor,
    keep_count: int,
    config: PyramidSinkKVConfig,
    layer_idx: int,
    attention: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, str, bool]:
    """Select ordered token indices for one layer.

    Sink and recent regions are always retained first.  Middle tokens are then
    selected by random, key-norm, legacy attention, or SnapKV attention scores.
    The returned indices are sorted so the compressed cache preserves original
    token order.
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
        _save_selection_debug(key, selected, keep_count, config, layer_idx, attention, config.score_method)
        return selected, config.score_method, False
    if remaining >= candidates.numel():
        selected = torch.cat([mandatory, candidates])
        selected = torch.unique(selected, sorted=True)
        assert_valid_selected_indices(selected, seq_len)
        _save_selection_debug(key, selected, keep_count, config, layer_idx, attention, config.score_method)
        return selected, config.score_method, False

    middle, method, fallback = _select_middle_by_method(key, candidates, remaining, config.score_method, config, layer_idx, attention)

    if middle.ndim == 1:
        selected = torch.cat([mandatory, middle])
        selected = torch.unique(selected, sorted=True)
    elif middle.ndim == 2:
        mandatory_rows = mandatory.view(1, -1).expand(middle.shape[0], -1)
        selected = torch.cat([mandatory_rows, middle], dim=-1)
        selected = torch.sort(selected, dim=-1).values
    else:
        raise ValueError(f"Expected 1D or 2D middle indices, got shape {tuple(middle.shape)}")
    assert_valid_selected_indices(selected, seq_len)
    _save_selection_debug(key, selected, keep_count, config, layer_idx, attention, method)
    return selected, method, fallback


def assert_valid_selected_indices(indices: torch.Tensor, seq_len: int) -> None:
    """Defensive checks for compressed KV gather indices."""

    if indices.ndim not in (1, 2):
        raise AssertionError(f"selected indices must be 1D or 2D, got {tuple(indices.shape)}")
    if indices.numel() == 0:
        raise AssertionError("selected indices must not be empty")
    if torch.any(indices < 0) or torch.any(indices >= seq_len):
        raise AssertionError(f"selected indices out of range [0, {seq_len})")
    if indices.ndim == 1 and indices.numel() > 1:
        diffs = indices[1:] - indices[:-1]
        if torch.any(diffs <= 0):
            raise AssertionError("selected indices must be sorted and unique")
    if indices.ndim == 2 and indices.shape[-1] > 1:
        diffs = indices[:, 1:] - indices[:, :-1]
        if torch.any(diffs <= 0):
            raise AssertionError("selected indices must be sorted and unique per head")


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
    fallback_layers = []

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
        indices, used_method, fallback = select_token_indices(key, target_keep, config, layer_idx, attention)

        compressed_key, compressed_value = gather_kv_by_indices(key, value, indices)
        compressed_seq_len = int(compressed_key.shape[-2])
        compressed_layers.append((compressed_key, compressed_value))
        if fallback:
            fallback_layers.append(layer_idx)
        per_layer.append(
            {
                "layer": layer_idx,
                "original_seq_len": seq_len,
                "compressed_seq_len": compressed_seq_len,
                "selected_index_shape": [int(x) for x in indices.shape],
                "per_head_selection": bool(indices.ndim == 2),
                "target_keep_ratio": float(ratio),
                "actual_keep_ratio": float(compressed_seq_len / max(seq_len, 1)),
                "score_method": used_method,
                "requested_score_method": config.score_method,
                "fallback_to_key_norm": bool(fallback),
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
        "per_layer_keep_lengths": [item["compressed_seq_len"] for item in per_layer],
        "score_method": config.score_method,
        "score_method_fallback": bool(fallback_layers),
        "fallback_score_method": "key_norm" if fallback_layers else None,
        "fallback_layers": fallback_layers,
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
    prefill_attention_mask = getattr(encoded, "attention_mask", torch.ones_like(input_ids)).to(device)
    logical_seq_len = int(input_ids.shape[1])
    output_attentions = (not dense) and config.score_method in ("attention", "snapkv")
    original_attn_impl = getattr(model.config, "_attn_implementation", None)
    if output_attentions and original_attn_impl is not None:
        model.config._attn_implementation = "eager"

    synchronize_if_cuda(device)
    start = time.perf_counter()
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=prefill_attention_mask,
            use_cache=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
        past_key_values = prefill_outputs.past_key_values
        compression_stats: Dict[str, Any] = {
            "achieved_compression_ratio": 1.0,
            "original_tokens": logical_seq_len * len(_cache_to_legacy(past_key_values)),
            "compressed_tokens": logical_seq_len * len(_cache_to_legacy(past_key_values)),
            "per_layer": [],
            "per_layer_keep_lengths": [],
            "score_method_fallback": False,
            "fallback_score_method": None,
            "scoring_overhead_sec": 0.0,
            "compression_overhead_sec": 0.0,
        }
        if not dense:
            compression_start = time.perf_counter()
            past_key_values, compression_stats = compress_past_key_values(
                past_key_values,
                config,
                getattr(prefill_outputs, "attentions", None),
            )
            compression_stats["compression_overhead_sec"] = time.perf_counter() - compression_start
            compression_stats["scoring_overhead_sec"] = 0.0
            if output_attentions:
                compression_stats["scoring_note"] = (
                    "SnapKV/attention scoring used attention weights returned by the prefill pass; "
                    "its overhead is included in TTFT rather than measured as a separate extra pass."
                )
                if getattr(model.config, "_attn_implementation", None) == "eager":
                    # Eager attention is needed for attention weights during
                    # prefill, but GPTNeoX eager decode assumes one global mask
                    # length and cannot handle layer-wise compressed KV lengths.
                    # Decode with the default SDPA path after scoring.
                    model.config._attn_implementation = "sdpa"
                    compression_stats["decode_attention_implementation"] = "sdpa"
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
        # A single decode attention_mask is valid only when all layers share the
        # same physical cache length.  Non-uniform budgets intentionally give
        # layers different lengths, so we omit the padding mask in that case and
        # keep RoPE correct with absolute position_ids/cache_position.
        attention_mask = decode_attention_mask_for_cache(past_key_values, device)
        synchronize_if_cuda(device)
        step_start = time.perf_counter()
        model_kwargs = {
            "input_ids": current_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "use_cache": True,
            "return_dict": True,
        }
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        with torch.no_grad():
            outputs = model(**model_kwargs)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        synchronize_if_cuda(device)
        decode_times.append(time.perf_counter() - step_start)
        logical_seq_len += 1
        generated.append(int(next_token.item()))

    total_time = ttft + sum(decode_times)
    if output_attentions and original_attn_impl is not None:
        model.config._attn_implementation = original_attn_impl
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
        "achieved_compression_ratio": compression_stats.get("achieved_compression_ratio"),
        "scoring_overhead_sec": compression_stats.get("scoring_overhead_sec", 0.0),
        "compression_overhead_sec": compression_stats.get("compression_overhead_sec", 0.0),
        "score_method_fallback": compression_stats.get("score_method_fallback", False),
        "fallback_score_method": compression_stats.get("fallback_score_method"),
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
        "achieved_compression_ratio": None,
        "scoring_overhead_sec": 0.0,
        "compression_overhead_sec": 0.0,
        "score_method_fallback": False,
        "fallback_score_method": None,
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
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    except (TypeError, ValueError) as exc:
        if "attn_implementation" not in kwargs:
            raise
        LOGGER.warning(
            "This transformers/model combination did not accept attn_implementation=%r; "
            "retrying without it. SnapKV will fall back to key_norm if attention weights remain unavailable.",
            attn_implementation,
        )
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.to(resolved_device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, resolved_device
