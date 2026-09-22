# 实验代码

这些改动没有进入最终线上 BF16 版本。修改独立 runtime 后再启动测试，勿直接覆盖运行中服务。

| 实验 | 实施方式 | 结果 |
|---|---|---|
| QK Norm + MRoPE | 把 `fused_qk_mrope.py` 复制为 runtime overlay 中的 `sglang/srt/layers/fused_qk_mrope_local.py`，再应用 `patches/qk-fusion.patch` | decode 11.79 ms/token，数值存在 BF16 舍入差异 |
| Piecewise graph | 应用 `patches/piecewise-static-shape.patch`，以独立配置启用 piecewise buckets [1,32,128,512,1024,2048,4096]，不要同时启用本版本的 torch.compile 开关 | 可运行但 TTFT 回退，未采用 |
| FP8 | 应用 `patches/fp8-scale-layout.patch`，加入 `--quantization fp8` | 8 并发 680.7 token/s，但质量评估不足，且该实验没有叠加 compile/IPC |

上述三个实验补丁的路径没有 `a/` 前缀，在独立 overlay 根目录用 `patch --dry-run -p0 < PATCH` 预检查，然后使用 `patch -p0`。DP Attention 补丁使用 `a/` 前缀，需 `-p1`。

QK 数值测试需设置 `MODEL_RUNTIME`，在 `experiments/` 下执行 `test_fused_qk_mrope.py`。它依赖 SGLang、FlashInfer、Triton 和可用 CUDA GPU。

参考 [完整实施报告](../docs/optimization-report.md)，原始实验数据在 `benchmarks/results/`。小规模正确性通过不等于业务质量无损。
