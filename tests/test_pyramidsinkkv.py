import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyramidsinkkv import (
    PyramidSinkKVConfig,
    _snapkv_attention_scores,
    cache_seq_length,
    compress_past_key_values,
    gather_kv_by_indices,
    layer_keep_ratios,
    select_token_indices,
)


def test_layer_ratios_keep_average_close():
    ratios = layer_keep_ratios(6, 0.5, "pyramid")
    assert ratios[0] > ratios[-1]
    assert abs(sum(ratios) / len(ratios) - 0.5) < 0.02

    reversed_ratios = layer_keep_ratios(6, 0.5, "reversed")
    assert reversed_ratios[0] < reversed_ratios[-1]
    assert abs(sum(reversed_ratios) / len(reversed_ratios) - 0.5) < 0.02

    spindle_ratios = layer_keep_ratios(6, 0.5, "spindle")
    assert max(spindle_ratios[2:4]) > max(spindle_ratios[0], spindle_ratios[-1])
    assert abs(sum(spindle_ratios) / len(spindle_ratios) - 0.5) < 0.02

    hourglass_ratios = layer_keep_ratios(6, 0.5, "hourglass")
    assert min(hourglass_ratios[2:4]) < min(hourglass_ratios[0], hourglass_ratios[-1])
    assert abs(sum(hourglass_ratios) / len(hourglass_ratios) - 0.5) < 0.02


def test_compress_cache_preserves_shape_and_logical_length_can_differ():
    torch.manual_seed(0)
    layers = []
    for _ in range(3):
        key = torch.randn(1, 2, 32, 4)
        value = torch.randn(1, 2, 32, 4)
        layers.append((key, value))

    config = PyramidSinkKVConfig(
        compression_ratio=0.5,
        sink_size=2,
        recent_size=4,
        budget_mode="pyramid",
        score_method="key_norm",
    )
    compressed, stats = compress_past_key_values(tuple(layers), config)

    assert len(compressed) == 3
    assert compressed[0][0].shape[:2] == (1, 2)
    assert compressed[0][0].shape[-1] == 4
    assert stats["achieved_compression_ratio"] < 1.0

    logical_seq_len = 32
    compressed_cache_len = cache_seq_length(compressed)
    assert compressed_cache_len != logical_seq_len
    assert compressed_cache_len < logical_seq_len


def test_selected_indices_sorted_unique_and_preserve_sink_recent():
    key = torch.randn(1, 2, 16, 4)
    attention = torch.zeros(1, 2, 16, 16)
    attention[:, :, -4:, 7] = 10.0
    config = PyramidSinkKVConfig(
        compression_ratio=0.5,
        sink_size=2,
        recent_size=3,
        budget_mode="uniform",
        score_method="snapkv",
        observation_window=4,
    )

    indices, used_method, fallback = select_token_indices(key, 8, config, layer_idx=0, attention=attention)

    assert used_method == "snapkv"
    assert fallback is False
    assert indices.tolist() == sorted(set(indices.tolist()))
    assert {0, 1}.issubset(set(indices.tolist()))
    assert {13, 14, 15}.issubset(set(indices.tolist()))
    assert 7 in set(indices.tolist())


def test_gather_kv_length_matches_selected_indices():
    key = torch.randn(1, 2, 10, 4)
    value = torch.randn(1, 2, 10, 4)
    selected = torch.tensor([0, 2, 5, 9], dtype=torch.long)

    compressed_key, compressed_value = gather_kv_by_indices(key, value, selected)

    assert compressed_key.shape == (1, 2, 4, 4)
    assert compressed_value.shape == (1, 2, 4, 4)
    assert torch.equal(compressed_key[:, :, 1, :], key[:, :, 2, :])


def test_snapkv_score_shape_is_seq_len():
    attention = torch.rand(2, 3, 6, 12)
    scores = _snapkv_attention_scores(attention, seq_len=12, observation_window=4)

    assert scores is not None
    assert scores.shape == (12,)


def test_snapkv_pooling_spreads_local_importance():
    attention = torch.zeros(1, 1, 4, 8)
    attention[:, :, -2:, 4] = 9.0

    raw_scores = _snapkv_attention_scores(attention, seq_len=8, observation_window=2, pooling_kernel=1)
    pooled_scores = _snapkv_attention_scores(attention, seq_len=8, observation_window=2, pooling_kernel=3)

    assert raw_scores is not None
    assert pooled_scores is not None
    assert raw_scores.argmax().item() == 4
    assert pooled_scores.shape == (8,)
    assert pooled_scores[3].item() > 0.0
    assert pooled_scores[4].item() > 0.0
    assert pooled_scores[5].item() > 0.0


def test_snapkv_head_aggregation_max_keeps_head_specific_signal():
    attention = torch.zeros(1, 2, 4, 8)
    attention[:, 0, -2:, 2] = 9.0
    attention[:, :, -2:, 6] = 5.0

    mean_scores = _snapkv_attention_scores(
        attention,
        seq_len=8,
        observation_window=2,
        head_aggregation="mean",
    )
    max_scores = _snapkv_attention_scores(
        attention,
        seq_len=8,
        observation_window=2,
        head_aggregation="max",
    )

    assert mean_scores is not None
    assert max_scores is not None
    assert mean_scores.argmax().item() == 6
    assert max_scores.argmax().item() == 2


def test_snapkv_fallback_to_key_norm_is_recorded_when_attentions_unavailable():
    layers = []
    for _ in range(2):
        key = torch.randn(1, 2, 16, 4)
        value = torch.randn(1, 2, 16, 4)
        layers.append((key, value))
    config = PyramidSinkKVConfig(
        compression_ratio=0.5,
        sink_size=2,
        recent_size=3,
        budget_mode="uniform",
        score_method="snapkv",
        observation_window=4,
    )

    _, stats = compress_past_key_values(tuple(layers), config, attentions=None)

    assert stats["score_method_fallback"] is True
    assert stats["fallback_score_method"] == "key_norm"
    assert all(item["fallback_to_key_norm"] for item in stats["per_layer"])


if __name__ == "__main__":
    test_layer_ratios_keep_average_close()
    test_compress_cache_preserves_shape_and_logical_length_can_differ()
    test_selected_indices_sorted_unique_and_preserve_sink_recent()
    test_gather_kv_length_matches_selected_indices()
    test_snapkv_score_shape_is_seq_len()
    test_snapkv_pooling_spreads_local_importance()
    test_snapkv_head_aggregation_max_keeps_head_specific_signal()
    test_snapkv_fallback_to_key_norm_is_recorded_when_attentions_unavailable()
