# 技术汇报

[Qwen3-VL-32B DP Attention 兼容性修复与推理性能优化](qwen3-vl-dp-attention-and-optimization.pptx)

20 页中文 PowerPoint，正文、表格、数据图表可编辑。内容覆盖三个工程版本、两处 DP 兼容性根因、回归测试、Profiler 方法、逐项优化、未采用方案以及最终验证边界。每页备注包含仓库内证据位置。

测试数据为 2026-09-17 的实测，报告整理于 2026-09-18。字体采用 PingFang SC；没有该字体的设备可能替换字体，演示前应确认换行。已渲染检查，未在 Microsoft PowerPoint 原生应用内测试。

柱状图从零开始，数值来源可在 `../benchmarks/results/` 复查。性能结果针对合成负载，不代表所有生产流量。
