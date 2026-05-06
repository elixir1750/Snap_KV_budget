# 分组实验表格

本文件汇总 `results/snapkv_ablation_20260506/` 中的真实运行结果。实验使用 `EleutherAI/pythia-70m`、WikiText test、`max_length=1024`、`stride=128`、`max_windows=2`、CPU float32。random baseline 的 PPL 使用 seeds `0,1,2,3,4` 报告均值和标准差；generation 指标为 3 次重复的均值和标准差。`key_norm` 不作为主实验方法展示，只在 `snapkv_fallback_status=fallback_to_key_norm` 时作为 fallback 说明出现；本次 SnapKV 结果均为 `true_snapkv`。

## 主实验表格

| method | budget_mode | score_method | sink_size | recent_size | compression_ratio | ppl | ttft | tpot | throughput | total_time | kv_cache_memory_mb | achieved_compression_ratio | scoring_overhead_sec | snapkv_fallback_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | dense | none | 4 | 64 | 1.0 | 21.1162 | 0.0671 +/- 0.0007 | 0.0049 +/- 0.0000 | 193.72 +/- 1.90 | 1.3217 +/- 0.0129 | 19.50 +/- 0.00 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | not_applicable | Dense KV-cache baseline |
| no_cache | no_cache | none | 0 | 0 | 1.0 | 21.1163 | 0.0717 +/- 0.0028 | 0.0854 +/- 0.0042 | 11.75 +/- 0.55 | 21.8452 +/- 1.0614 | 0.00 +/- 0.00 | TODO | 0.0000 +/- 0.0000 | not_applicable | Optional no-cache baseline |
| uniform random | uniform | random | 4 | 64 | 0.5 | 33.8358 +/- 2.0319 | 0.0712 +/- 0.0004 | 0.0046 +/- 0.0001 | 206.84 +/- 3.38 | 1.2380 +/- 0.0201 | 12.73 +/- 0.00 | 0.4991 +/- 0.0000 | 0.0000 +/- 0.0000 | not_applicable | PPL seeds=0,1,2,3,4 |
| uniform snapkv | uniform | snapkv | 4 | 64 | 0.5 | 40.3360 | 0.0835 +/- 0.0019 | 0.0052 +/- 0.0001 | 183.27 +/- 1.73 | 1.3970 +/- 0.0132 | 12.73 +/- 0.00 | 0.4991 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | SnapKV attention-based selection |
| pyramid random | pyramid | random | 4 | 64 | 0.5 | 40.0150 +/- 1.9032 | 0.0716 +/- 0.0002 | 0.0050 +/- 0.0003 | 191.56 +/- 10.49 | 1.3405 +/- 0.0762 | 12.78 +/- 0.00 | 0.5029 +/- 0.0000 | 0.0000 +/- 0.0000 | not_applicable | PPL seeds=0,1,2,3,4 |
| pyramid snapkv | pyramid | snapkv | 4 | 64 | 0.5 | 45.6161 | 0.0772 +/- 0.0031 | 0.0043 +/- 0.0002 | 217.02 +/- 8.35 | 1.1814 +/- 0.0466 | 12.78 +/- 0.00 | 0.5029 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | SnapKV attention-based selection |
| spindle random | spindle | random | 4 | 64 | 0.5 | 26.3768 +/- 2.1393 | 0.0693 +/- 0.0011 | 0.0043 +/- 0.0001 | 219.57 +/- 2.31 | 1.1660 +/- 0.0122 | 12.73 +/- 0.00 | 0.4997 +/- 0.0000 | 0.0000 +/- 0.0000 | not_applicable | PPL seeds=0,1,2,3,4 |
| spindle snapkv | spindle | snapkv | 4 | 64 | 0.5 | 26.4305 | 0.0782 +/- 0.0014 | 0.0043 +/- 0.0001 | 218.23 +/- 5.84 | 1.1739 +/- 0.0320 | 12.73 +/- 0.00 | 0.4997 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | SnapKV attention-based selection |
| reversed snapkv | reversed | snapkv | 4 | 64 | 0.5 | 39.7490 | 0.0840 +/- 0.0004 | 0.0049 +/- 0.0001 | 191.02 +/- 5.24 | 1.3412 +/- 0.0362 | 12.78 +/- 0.00 | 0.5029 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | Optional budget-shape ablation |
| hourglass snapkv | hourglass | snapkv | 4 | 64 | 0.5 | 58.8478 | 0.0809 +/- 0.0006 | 0.0046 +/- 0.0000 | 204.80 +/- 0.26 | 1.2500 +/- 0.0016 | 12.74 +/- 0.00 | 0.5003 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | Optional budget-shape ablation |

## Compression Ratio Sweep 表格

| method | budget_mode | score_method | compression_ratio | ppl | ttft | tpot | throughput | total_time | kv_cache_memory_mb | achieved_compression_ratio | scoring_overhead_sec | snapkv_fallback_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | dense | none | 1.0 | 21.1162 | 0.0778 +/- 0.0076 | 0.0051 +/- 0.0004 | 187.60 +/- 13.35 | 1.3719 +/- 0.1021 | 19.50 +/- 0.00 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | not_applicable | Dense KV-cache baseline |
| uniform random | uniform | random | 0.75 | 25.8829 +/- 2.3710 | 0.0675 +/- 0.0008 | 0.0044 +/- 0.0002 | 214.17 +/- 7.77 | 1.1969 +/- 0.0431 | 16.12 +/- 0.00 | 0.7504 +/- 0.0000 | 0.0000 +/- 0.0000 | not_applicable | PPL seeds=0,1,2,3,4 |
| uniform snapkv | uniform | snapkv | 0.75 | 29.5593 | 0.0784 +/- 0.0018 | 0.0048 +/- 0.0001 | 197.62 +/- 4.78 | 1.2961 +/- 0.0309 | 16.12 +/- 0.00 | 0.7504 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | SnapKV attention-based selection |
| spindle random | spindle | random | 0.75 | 22.5578 +/- 0.1885 | 0.0694 +/- 0.0010 | 0.0047 +/- 0.0003 | 202.80 +/- 12.52 | 1.2674 +/- 0.0818 | 16.12 +/- 0.00 | 0.7499 +/- 0.0000 | 0.0000 +/- 0.0000 | not_applicable | PPL seeds=0,1,2,3,4 |
| spindle snapkv | spindle | snapkv | 0.75 | 23.8238 | 0.0860 +/- 0.0050 | 0.0048 +/- 0.0002 | 195.26 +/- 6.07 | 1.3124 +/- 0.0413 | 16.12 +/- 0.00 | 0.7499 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | SnapKV attention-based selection |
| uniform random | uniform | random | 0.5 | 33.8358 +/- 2.0319 | 0.0666 +/- 0.0011 | 0.0043 +/- 0.0000 | 220.94 +/- 1.00 | 1.1587 +/- 0.0053 | 12.73 +/- 0.00 | 0.4991 +/- 0.0000 | 0.0000 +/- 0.0000 | not_applicable | PPL seeds=0,1,2,3,4 |
| uniform snapkv | uniform | snapkv | 0.5 | 40.3360 | 0.0773 +/- 0.0015 | 0.0046 +/- 0.0001 | 203.00 +/- 4.49 | 1.2617 +/- 0.0275 | 12.73 +/- 0.00 | 0.4991 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | SnapKV attention-based selection |
| spindle random | spindle | random | 0.5 | 26.3768 +/- 2.1393 | 0.0712 +/- 0.0026 | 0.0048 +/- 0.0002 | 198.34 +/- 9.48 | 1.2936 +/- 0.0602 | 12.73 +/- 0.00 | 0.4997 +/- 0.0000 | 0.0000 +/- 0.0000 | not_applicable | PPL seeds=0,1,2,3,4 |
| spindle snapkv | spindle | snapkv | 0.5 | 26.4305 | 0.0860 +/- 0.0087 | 0.0048 +/- 0.0001 | 197.40 +/- 4.11 | 1.2975 +/- 0.0273 | 12.73 +/- 0.00 | 0.4997 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | SnapKV attention-based selection |
| uniform random | uniform | random | 0.25 | 50.5768 +/- 1.8445 | 0.0727 +/- 0.0026 | 0.0045 +/- 0.0002 | 211.14 +/- 7.74 | 1.2141 +/- 0.0446 | 9.35 +/- 0.00 | 0.2496 +/- 0.0000 | 0.0000 +/- 0.0000 | not_applicable | PPL seeds=0,1,2,3,4 |
| uniform snapkv | uniform | snapkv | 0.25 | 58.2325 | 0.0864 +/- 0.0008 | 0.0049 +/- 0.0002 | 190.67 +/- 7.26 | 1.3446 +/- 0.0501 | 9.35 +/- 0.00 | 0.2496 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | SnapKV attention-based selection |
| spindle random | spindle | random | 0.25 | 41.0463 +/- 1.2718 | 0.0773 +/- 0.0028 | 0.0051 +/- 0.0002 | 185.41 +/- 6.58 | 1.3825 +/- 0.0503 | 9.62 +/- 0.00 | 0.2692 +/- 0.0000 | 0.0000 +/- 0.0000 | not_applicable | PPL seeds=0,1,2,3,4 |
| spindle snapkv | spindle | snapkv | 0.25 | 47.7264 | 0.0884 +/- 0.0067 | 0.0047 +/- 0.0002 | 200.22 +/- 10.88 | 1.2824 +/- 0.0689 | 9.62 +/- 0.00 | 0.2692 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | SnapKV attention-based selection |

## Sink/Recent 消融表格

| method | budget_mode | score_method | sink_size | recent_size | compression_ratio | ppl | ttft | tpot | throughput | total_time | kv_cache_memory_mb | achieved_compression_ratio | scoring_overhead_sec | snapkv_fallback_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spindle snapkv sink recent | spindle | snapkv | 4 | 64 | 0.5 | 26.4305 | 0.0809 +/- 0.0031 | 0.0048 +/- 0.0001 | 194.41 +/- 4.28 | 1.3175 +/- 0.0290 | 12.73 +/- 0.00 | 0.4997 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | sink + recent |
| spindle snapkv no sink | spindle | snapkv | 0 | 64 | 0.5 | 26.2927 | 0.0825 +/- 0.0027 | 0.0047 +/- 0.0002 | 198.18 +/- 6.29 | 1.2930 +/- 0.0402 | 12.73 +/- 0.00 | 0.4997 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | no sink |
| spindle snapkv no recent | spindle | snapkv | 4 | 0 | 0.5 | 26.2726 | 0.0840 +/- 0.0019 | 0.0049 +/- 0.0001 | 193.85 +/- 4.57 | 1.3213 +/- 0.0307 | 12.73 +/- 0.00 | 0.4997 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | no recent |
| spindle snapkv no sink no recent | spindle | snapkv | 0 | 0 | 0.5 | 26.2553 | 0.0833 +/- 0.0026 | 0.0048 +/- 0.0002 | 197.28 +/- 6.82 | 1.2992 +/- 0.0441 | 12.73 +/- 0.00 | 0.4997 +/- 0.0000 | 0.0000 +/- 0.0000 | true_snapkv | no sink + no recent |
