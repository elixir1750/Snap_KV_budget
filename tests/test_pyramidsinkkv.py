import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyramidsinkkv import PyramidSinkKVConfig, cache_seq_length, compress_past_key_values, layer_keep_ratios


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


if __name__ == "__main__":
    test_layer_ratios_keep_average_close()
    test_compress_cache_preserves_shape_and_logical_length_can_differ()
