# 第三课：Profiler、配置消融和证据分析：参考答案

对照顺序：B(TP2)对A(DP2)，C(compile)对B，D(通信融合)对B，E(compile+融合)对C，F(keep-device)对C，G(显式IPC)对F。最终组合H另做端到端验证。这样才能区分单项和交互效应。

无GPU活动比例：给定窗口[a,b]，将kernel、memcpy、memset区间裁剪后取并集U，计算 `1 - U/(b-a)`。例如[0,10]和[5,15]并集为15，不能用相加得到20。这个指标不是SM空闲率，也不等于HBM利用率。

归档trace重新解析显示：普通TP2短文本每步kernel中位数1353，compile+融合1158。显式IPC图像scheduler trace不再包含24MiB HtoD，但存在P2P或设备内部复制，因此不能说全链路零拷贝。所有图像TTFT收益也不能只归给一次DMA，因为跨进程交付路径同时变化。

结果表由原始逐请求JSON重算，见results/candidate_metrics.csv。trace_summary有各rank与阶段，不能用两个GPU的kernel耗时之和代替请求延迟。候选历史数据通常只有3轮，不能冒充课程要求的新增5轮实测。

## 执行与证据

在包根目录运行：

```bash
.venv-local/bin/python 03_profiling/verify.py
```

本次重解析12份历史GPU trace，核对8个配置的基准数据，并独立检查5组区间并集样例及原始memcpy事件。
配置生成：`python3 gpu_lab.py --stage ablation --plan`。
新GPU采集：去掉`--plan`并指定模型目录，输出在独立gpu_runs目录。

本地CPU/归档检查与新GPU验证分别记录，详见下节。

## 新增 GPU 实测与验收

已完成A/B/C/D/E/F/G/Q/H九配置无profiler基准，每类5轮，完整指标与TPOT范围见results/fresh-gpu/candidate_metrics.csv。B/E/G短文本及图像各采双rank trace，共12份。逐kernel统计、区间并集和24MiB拷贝核验见results/fresh-gpu/trace_analysis与memcpy_evidence.json。

B每步kernel中位数1353，E为1158。图像B含24MiB HtoD，G未出现该拷贝，仍应查看其他P2P/设备内复制。profiler下短文本间隔受测量扰动，判断服务收益使用无profiler基准。完整本次表见FRESH_GPU_REPORT.md。
