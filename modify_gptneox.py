"""Compatibility helpers for older SnapKV-style imports.

The current PyramidSinkKV implementation does not monkey-patch GPTNeoX modules.
It compresses ``past_key_values`` after prefill and then decodes with explicit
absolute ``position_ids``/``cache_position``.  Keeping generation outside the
model patch makes the RoPE behavior easier to audit for a course project.
"""

from pyramidsinkkv import PyramidSinkKVConfig, compress_past_key_values


def apply_snapkv_to_model(model, *_, **__):
    print(
        "apply_snapkv_to_model is deprecated. Use pyramidsinkkv.generate(...) or "
        "scripts/benchmark_generation.py so RoPE position ids are managed explicitly."
    )
    return model


def apply_pyramidsinkkv_to_model(model, *_, **__):
    return apply_snapkv_to_model(model)
