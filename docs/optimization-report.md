# Qwen3-VL-32B 优化实施与复测报告

2026-09-17 · 已实际修改、运行、对比并部署 · SGLang 0.5.6.post2 / torch 2.9.1+cu128 / H200 ×2

**主服务已切换为 BF16、普通 TP2、torch.compile、FlashInfer all-reduce 融合和多模态 CUDA IPC。** 端口、模型名、鉴权、32K 上下文和全局最大并发 8 保持一致。DP Attention 已关闭，因为当前实测负载下普通 TP2 更快。FP8、自定义 QK 融合和 piecewise graph 均实际运行过，其中前两项保留为实验候选，piecewise 因延迟回退未采用。

![优化前后](../evidence/optimization/优化前后.png)

## 1. 主服务优化前后

下表使用同一主服务 GPU 0/1、切换前后各 5 轮无 profiler 测量；单请求指标 n=5，8 并发指标 n=40。变化按中位数或总吞吐计算。

| 指标 | 优化前 | 部署后 | 变化 |
|---|---:|---:|---:|
| 短文本首 token | 71.56 ms | 52.91 ms | -26.1% |
| 短文本 decode | 15.03 ms/token | 11.25 ms/token | -25.1% |
| 约 4K 文本首 token | 322.50 ms | 252.07 ms | -21.8% |
| 1024×1024 单图首 token | 254.87 ms | 110.92 ms | -56.5% |
| 8 并发输出吞吐 | 447.48 token/s | 572.04 token/s | +27.8% |
| 8 并发首 token | 269.60 ms | 265.41 ms | -1.6% |


这是合成文本和几何图负载的结果，不是生产流量 P99。8 并发首 token 不一定随 decode 加速同幅改善：prefill、请求入队和批处理仍在关键路径上。

逐请求原始结果：[before_cutover-baseline.json](../evidence/optimization/before_cutover-baseline.json)、[deployed-baseline.json](../evidence/optimization/deployed-baseline.json)；全部汇总：[benchmark-summary.json](../evidence/optimization/benchmark-summary.json)。

## 2. 本次实际执行的优化

| 手段 | 实际操作与证据 | 最终处理 |
|---|---|---|
| 并行布局与冗余计算 | DP2 Attention 改为普通 TP2；保持 chunked prefill=4096、max-running=8、graph max batch=8，消除 DP 布局转换及 idle rank attention 的额外工作 | 已部署 |
| torch.compile | 编译 decode 路径，捕获 batch 1/2/4/8；实际预热、基准与 CPU/CUDA trace | 已部署 |
| 通信/算子融合 | 开启 `--enable-flashinfer-allreduce-fusion`；trace 确认 fusion kernel 执行，单独开启与 compile 组合分别测试 | 已部署 |
| 图文 pipeline / 冗余搬运 | 对照仅 keep-device 与 keep-device + 显式 CUDA IPC；后者显著降低单图 TTFT，并消除 scheduler 侧 24 MiB H→D | 已部署 |
| 前缀缓存 | 普通 TP2 使用同一个逻辑请求队列，重复长前缀检查命中收益；保留框架 Radix Cache | 已验证；未另造缓存实现 |
| piecewise CUDA Graph | 修复静态分区 `sym_shape_indices` 为空时的索引错误；捕获 token buckets 1/32/128/512/1024/2048/4096，执行完整基准与 19 项功能检查 | 性能回退，未部署 |
| Q/K RMSNorm + MRoPE 融合 | 编写并接入 Triton kernel；修复把 token 数设为 constexpr 导致的反复编译；跑 16 组数值/微基准、19 项图文与 16 项质量检查 | 有收益，但存在 BF16 舍入差异，保留实验 |
| FP8 权重/GEMM | 在线 FP8 实际启动失败后，定位并修复视觉 MLP 缩放布局；重启成功，完成三轮性能和 35 项小规模正确性检查 | 精度评估不足，保留实验 |

### 所有候选的实测数据

| 实际运行配置 | 短文本 TTFT ms | decode ms/token | 单图 TTFT ms | 8 并发 token/s |
|---|---:|---:|---:|---:|
| 原 DP2 Attention | 69.2 | 15.07 | 254.5 | 447.0 |
| 普通 TP2 | 54.5 | 12.35 | 265.0 | 521.0 |
| TP2 + compile | 58.1 | 11.88 | 252.2 | 569.2 |
| TP2 + 通信融合 | 53.5 | 12.23 | 251.2 | 532.3 |
| TP2 + compile + 通信融合 | 65.1 | 11.29 | 264.5 | 570.7 |
| TP2 + compile + keep-device | 53.9 | 11.86 | 289.7 | 570.0 |
| TP2 + compile + keep-device + 显式 IPC | 49.6 | 11.84 | 109.2 | 569.1 |
| TP2 + piecewise graph（修复后） | 76.4 | 12.35 | 322.3 | 518.0 |
| TP2 + 自定义 QK Norm/MRoPE 融合 | 51.8 | 11.79 | 254.8 | 559.0 |
| TP2 + 在线 FP8（布局修复后） | 44.8 | 9.44 | 332.1 | 680.7 |
| 完整 BF16 组合，隔离实例 | 52.8 | 11.25 | 116.4 | 572.0 |
| 完整 BF16 组合，主服务 | 52.9 | 11.25 | 110.9 | 572.0 |


候选通常 3 轮；原 DP2 首次基线、切换前基线、完整组合和最终主服务均为 5 轮。候选在独立 GPU 对上运行，部分初始化/验证工作会同时使用宿主 CPU；最后主服务测试时临时模型服务均停止。未锁定 GPU 时钟，不能把 1% 左右差异当作显著结论。通信融合单独开启收益很小，但与 compile 组合后单请求 decode 进一步下降；8 并发吞吐相对 compile 单独开启基本持平。

只设置 `--keep-mm-feature-on-device` 没有稳定图像收益。显式设置 `SGLANG_USE_CUDA_IPC_TRANSPORT=1` 才是本次 pipeline 改善的关键。

## 3. Profiler 证据：算子、图融合与运行空泡

实际采集 `torch.profiler` CPU + CUDA，record_shapes=true，32 个输出 token。按 `cudaGraphLaunch` correlation 归属每步 kernel，丢弃最前 5、最后 2 步，统计稳定 25 步。

| 配置 / 阶段 | rank | 每步 kernel 中位数 | graph GPU 跨度 ms | decode 无 GPU 活动比例 | prefill/setup 无 GPU 活动比例 |
|---|---:|---:|---:|---:|---:|
| allreduce_fusion | 0 | 1289 | 11.95 | 14.5% | 49.3% |
| allreduce_fusion | 1 | 1289 | 11.95 | 14.2% | 27.1% |
| compile_allreduce | 0 | 1158 | 11.12 | 14.3% | 44.9% |
| compile_allreduce | 1 | 1158 | 11.11 | 14.4% | 28.9% |
| deployed | 0 | 1158 | 11.09 | 15.3% | 46.3% |
| deployed | 1 | 1158 | 11.11 | 15.0% | 15.4% |
| qk_fused_dynamic | 0 | 1033 | 11.56 | 12.8% | 15.7% |
| qk_fused_dynamic | 1 | 1033 | 11.52 | 13.7% | 28.2% |


上轮同版本、同模型普通 TP2 trace 每步 1353 个 kernel；通信融合为 1289；compile + 通信融合为 1158。原 DP Attention trace 为 1620。内核数量减少并不等于同等比例的延迟下降：剩余 GEMM 仍占主要时间，融合还受到访存、同步及 batch shape 影响。

“无 GPU 活动比例”是窗口中 kernel/memcpy/memset 区间并集之外的比例，并非 SM 空闲率。各 trace 的 CPU profiling 开销、rank 起始窗口不同，不能将百分比直接当线上空泡的精确改善。**本次已降低实际 decode 延迟，但没有消除 prefill 的 CPU dispatch 和调度间隙。** 未采集 Nsight Compute 的 HBM/SM 计数器，不能声称已证实带宽饱和。

最终主服务短文本 decode 中，按 kernel 名称分类，compiler_fused 占两卡 kernel 时间和约 82.8% / 81.3%，通信约 5.8% / 7.2%，Attention 约 6.0%。compile 后的混合算子类别不能直接与原始 GEMM 占比一一比较。上轮基准与 trace 见 [前次性能分析](../evidence/profiling/性能分析报告.md)。

原始 trace、逐 kernel 统计和分类结果：[analysis/trace-summary.json](../evidence/optimization/analysis/trace-summary.json)、[analysis/kernels.csv](../evidence/optimization/analysis/kernels.csv)、[trace-index.md](../evidence/optimization/trace-index.md)。自定义融合 kernel 按名称归类仅用于展示，不能将整段 fused 时间归于单一原始算子。

### 24 MiB 图像搬运的变化

上轮普通 TP2 scheduler trace，每个 rank 的 H→D 总量约 25.43 MB，其中一个 25,165,824-byte（24 MiB）拷贝来自展开的图像 patch 输入。

显式 IPC 后，同样 1024 方图：每个 rank 的 H→D 总量降到 264,012 bytes，24 MiB H→D 不再出现；rank 1 有一个 24 MiB P2P 拷贝，带 profiler 约 239 μs。rank 0 仍有设备内部复制。因此优化是**避免经主机回传/再次上传**，并非整个图像路径零拷贝。scheduler trace 直接证明的是 H→D 被替代；前端 D→H 的去除还依据 keep-device/IPC 执行路径，不把未重新采集的前端阶段宣称为本轮完整 trace 证据。

24 MiB 对象是 processor 展开的 patch 输入，不是宣称缓存了 ViT 的最终 embedding。图像首 token 减少的全部时间也不能只归给这一次 DMA：跨进程序列化和交付路径同时改变。本次没有建立全进程统一 request span。

传输统计：[analysis/memcpy-summary.json](../evidence/optimization/analysis/memcpy-summary.json)。IPC 配置增加了 GPU 侧缓存/传输池，候选启动日志中 rank 0 可用显存比无 IPC 少约 1.5 GiB，KV token 容量从 422,433 降到 415,005；当前 32K/8 请求配置有余量，不能无限放大图像并发。

## 4. 已写代码但未默认上线的方案

### Q/K Norm + MRoPE Triton kernel

融合连续化读取、Q/K RMSNorm、对应权重缩放和 MRoPE；限定 BF16、head_dim=128、Neox/interleaved MRoPE [24,20,20]，其他路径回退原实现。**没有融合 KV cache 写入**，也没有修改 Attention 算法。

TP2 的 Q/K heads=32/4，单 token 微基准从 9.55 μs 降至 1.71 μs；4050 tokens 从 263.8 μs 降至 90.9 μs。整模型收益远小于微基准倍数。最初将 token 数设为 constexpr，使 nonce 导致的新 prompt shape 反复触发编译，短文本 TTFT 143 ms；改为运行时参数后约 52 ms。这一修复已实际复测。

16 组随机张量测试最大绝对差 0.03125，最少逐元素完全相等比例 78.8%；通过的容差为 atol=0.03125、rtol=0.015625，**不是逐位等价**。19 项图文与 16 项算术/OCR检查通过，不能替代真实业务精度评估。主服务采用成熟 compile/通信融合路径，没有加载此自定义 kernel。

代码：[scripts/fused_qk_mrope.py](../evidence/optimization/scripts/fused_qk_mrope.py)、[qk-fusion.patch](../evidence/optimization/qk-fusion.patch)；数值与微基准：[fused-kernel-test.json](../evidence/optimization/fused-kernel-test.json)。

### Piecewise graph

启动先遇到静态分区没有符号 shape、直接索引 `sym_shape_indices[0]` 的错误。对这些分区回退通用编译图后可启动，19 项图文检查通过，但短文本 TTFT 76.4 ms、单图 322.3 ms，均差于普通 TP2；decode 基本无收益。padding 分桶与静态分区回退可能影响收益，但没有逐项归因实验，不把它们写成已证实的唯一根因。当前版本的这组配置不采用。

补丁与失败记录：[piecewise-static-shape.patch](../evidence/optimization/piecewise-static-shape.patch)、[piecewise-initial-failure.txt](../evidence/optimization/piecewise-initial-failure.txt)。

### FP8

`--quantization fp8` 对 BF16 权重在线量化。初次启动在视觉 MLP 的 Triton fallback 断言失败：在线加载器产生 `[1,N]` per-channel scale，而该 kernel 要求 `[N,1]`。适配仅 reshape 缩放布局，不改变 scale 数值；3 组小矩阵测试两种布局结果逐位相同，与反量化 FP32 reference 最大绝对差不超过 0.00098。

修复后每 rank 加载权重占用由约 31.34 GiB 降至 17.49 GiB，短文本 decode 9.44 ms/token、8 并发 680.7 token/s；但未叠加 compile/IPC，图像 TTFT 332.1 ms 比最终 BF16 方案差。35 项小规模检查通过，但没有业务数据集上的语言、OCR、文档、定位精度评估。因此不默认切换精度。没有把它与 BF16 收益简单相乘，也未声称 FP8 + 全部优化的组合已测。

补丁：[fp8-scale-layout.patch](../evidence/optimization/fp8-scale-layout.patch)；错误与验证：[fp8-startup-failure.txt](../evidence/optimization/fp8-startup-failure.txt)、[fp8-scale-test.json](../evidence/optimization/fp8-scale-test.json)、[fp8-fixed-quality.json](../evidence/optimization/fp8-fixed-quality.json)。

## 5. 正确性、缓存与服务状态

最终主服务验证包含：

- 健康检查、模型 ID、未鉴权返回 401、文本、图像和流式响应。
- 19 项混合文本/多图/不同尺寸/不同长度请求，最多 8 并发。
- 16 项算术、长文本复制和 OCR 精确答案检查。
- 72 张不同的 1024 方图，以 8 并发重复使用 IPC 缓存/传输池；全部背景颜色识别正确，完成时间 7.42 秒。这是生命周期和串图回归，不是固定输出吞吐基准。
- 30,021 token 文本、18,125 token 图文；视频从红色变到蓝色，均正确。
- 相同长前缀连续 6 次：完整请求耗时 282.7 ms → 后五次中位数 93.3 ms。这里是短答案的**完整 HTTP 请求耗时**，不是流式 TTFT；没有把 KV 命中宣称为自动跳过 ViT。

对应材料：[deployed-smoke.json](../evidence/optimization/deployed-smoke.json)、[deployed-functional.json](../evidence/optimization/deployed-functional.json)、[deployed-quality.json](../evidence/optimization/deployed-quality.json)、[deployed-pipeline.json](../evidence/optimization/deployed-pipeline.json)、[deployed-long-context.json](../evidence/optimization/deployed-long-context.json)。

服务运行于 GPU 0/1；所有临时实例均已停止，GPU 2–7 的实验模型进程已退出。主服务保留原先 DP 兼容 overlay，但当前配置不启用 DP Attention；本次实验的 QK/FP8/piecewise 源码补丁没有复制到主服务 overlay。修改的主服务文件是 config.json 和 service.py（增加从配置读取环境变量），并复制已验证的编译缓存以加速预热。

当前配置：[configs/deployed.json](../evidence/optimization/configs/deployed.json)；部署记录：[deployment.json](../evidence/optimization/deployment.json)。切换与 smoke 共耗时 182.9 秒；这不是每次启动的固定 SLA。

## 6. 复现、回滚与边界

主服务核心配置：

```text
--dtype bfloat16 --tp-size 2
--context-length 32768 --max-running-requests 8
--mem-fraction-static 0.60 --chunked-prefill-size 4096
--cuda-graph-max-bs 8 --enable-torch-compile --torch-compile-max-bs 8
--enable-flashinfer-allreduce-fusion --keep-mm-feature-on-device
SGLANG_USE_CUDA_IPC_TRANSPORT=1
SGLANG_MM_FEATURE_CACHE_MB=1024
```

远程根目录 `/deployment/qwen3-vl-32b`，实验数据在 `optimization/`。API Key 仍从原位置读取，没有写入本报告或测试数据。原配置与启动器保存于 `optimization/before/`。

基准与 profiler 复现（在远程部署目录执行；profiling 应与性能基准分开，线上高负载时不要同时采集）：

```bash
venv/bin/python optimization/opt_probe.py --port 30000 --label rerun --rounds 5
venv/bin/python optimization/opt_probe.py --port 30000 --label rerun --profile --cases short,image
```

回滚本次配置：

```bash
venv/bin/python service.py stop
cp optimization/before/config.json config.json
cp optimization/before/service.py service.py
venv/bin/python service.py start
```

测量方法：temperature=0、ignore_eos=true、固定输出 128 token，客户端在服务器本机；输入带随机 nonce 降低前缀命中，内容长度会有少量波动。TTFT 从发送请求到首个非空流式 chunk；本轮 TPOT 统一为 `(完整 HTTP 结束时间－首个非空 chunk 时间)/(输出 token数－1)`，避免不可见 token 导致旧的“最后可见 chunk”口径低估。汇总脚本从原始 elapsed_s/TTFT/usage 重算所有候选，包含末尾协议开销，不是硬件每 token 计时。输出吞吐包含 prefill 和批次完成耗时；图像 batch wall time还包含构造测试图片的客户端开销，各候选一致。5 轮不足以报告可靠 P95/P99。

未执行或未上线的更大改造：ViT 专项布局/融合、prefill/decode 独立部署、KV FP8、投机解码，以及按真实业务调整图片分辨率。它们需要额外实现、代表性数据或新的延迟/质量目标；本次不把降低分辨率造成的信息损失当作无代价加速。并发上限也保持 8，未用增加排队和 batch 的方式包装当前延迟收益。

后续最有价值的工作是：用真实业务样本评估 FP8 与自定义 QK 融合的质量，再分别测试它们与当前组合的交互；针对剩余 GEMM 采集硬件计数器；为前处理、IPC、ViT、prefill 和首 token 增加统一 request-id span。上述均为后续建议，未标为本次已实现。
