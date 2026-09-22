# Qwen3-VL-32B：DP Attention 修复与逐步性能优化

## 四次课程与 2026-09-22 实测

- [四课作业、独立答案目录与复现入口](courses/qwen-vl-course-answers/README.md)
- [九组配置的 GPU 实测报告](courses/qwen-vl-course-answers/FRESH_GPU_REPORT.md)
- [实验脚本修复与验证边界](courses/qwen-vl-course-answers/REPRODUCTION_NOTES.md)
- [机器可读验证记录](courses/qwen-vl-course-answers/verification/full-course-verification.json)

本次完成 9 组配置、495 条基准请求及 12 份新增 GPU trace。自定义 QK 内核发现 positions stride 缺陷，已拒绝部署；默认组合课程题严格匹配 39/40，1 题为 OCR 数字间空格差异。全部原始结果与失败证据随答案目录提供。


本项目保存 **Qwen3-VL-32B-Instruct 在 SGLang 0.5.6.post2 上的部署工程、兼容性修复、性能优化过程与实测证据**。测试使用两张 H200 和 NVLink，模型保持 BF16。实际推理框架是 **SGLang，不是 vLLM**。

最终选择普通 TP2、torch.compile、FlashInfer all-reduce 融合、多模态 CUDA IPC。DP Attention 修复版完整保留，但性能优化版关闭 DP Attention。**“支持 DP Attention”与“当前负载适合 DP Attention”是两个分别验证的问题。**

## 最终结果

2026-09-17，同一主服务 GPU 0/1，切换前后各 5 轮，无 profiler，固定输出 128 token。

| 指标 | 优化前：修复后的 DP2 Attention | 最终优化版 | 变化 |
|---|---:|---:|---:|
| 短文本首 token | 71.56 ms | 52.91 ms | -26.1% |
| 短文本解码 | 15.03 ms/token | 11.25 ms/token | -25.1% |
| 约 4K 文本首 token | 322.50 ms | 252.07 ms | -21.8% |
| 1024×1024 单图首 token | 254.87 ms | 110.92 ms | -56.5% |
| 8 并发输出吞吐 | 447.48 token/s | 572.04 token/s | +27.8% |
| 8 并发首 token | 269.60 ms | 265.41 ms | -1.6% |

合成文本/几何图，单请求指标 n=5，并发指标 n=40。未锁 GPU 时钟，不能将 1% 差异或本表直接当作生产 P99 结论。[原始数据](benchmarks/results/) · [完整实施报告](docs/optimization-report.md) · [DP Attention 修复文档](docs/dp-attention-compatibility.md) · [技术汇报 PPT（最新版，103 页）](slides/qwen3-vl-dp-attention-and-optimization-v12-zhihu-references.pptx)

## 版本与工程结构

| 目录 | 对应状态 | 配置与源码 |
|---|---|---|
| `versions/original/` | 最初单卡部署，尚未修复 DP 兼容性 | TP1，BF16，显存比例 0.85；SGLang 上游原始相关文件 |
| `versions/dp-attention-fixed/` | 完成兼容性修复后的性能基线 | TP2 / DP2，DP Attention，显存比例 0.60 |
| `versions/optimized/` | 最终经验证的优化方案 | TP2 / DP1，compile、通信融合、CUDA IPC |
| `patches/` | 可检查的最小源代码差异 | DP 修复以及三项隔离实验补丁 |
| `tests/` | 真实 SGLang 方法回归和 API 回归 | MRoPE、视觉 dispatch、混合图文、长上下文 |
| `benchmarks/` | 可配置测试客户端、逐请求结果和 profiler 分析 | 基准、质量、IPC 生命周期检查 |
| `environment/` | 验证环境及完整 pip-freeze | Python 3.12、torch 2.9.1+cu128 |
| `docs/`、`slides/` | 分步骤文档和可编辑技术汇报 | 数据来源和适用边界随文档保留 |
| `evidence/` | 归档报告依赖、脱敏 traces 与历史脚本 | 历史脚本用于追溯，可运行入口优先用 scripts/benchmarks |

三份版本快照可直接比较，Git 提交记录按“原始部署、DP 修复、性能优化、文档与汇报”组织。源文件哈希与脱敏说明见 [provenance.json](versions/provenance.json)。模型路径和内部连接信息已改为占位符。仓库不包含密钥、权重、虚拟环境或编译缓存。

## 快速运行

需要 Linux x86_64、NVIDIA CUDA GPU、Python 3.12，以及本地完整模型权重目录。双卡实测硬件是 H200；其他显卡需重新评估显存和算子支持。默认模型 revision 为 `0cfaf48183f594c314753d30a4c4974bc75f3ccb`，权重单独获取。

```bash
git clone https://github.com/justsoloking-blip/qwen-vl-32B-project.git
cd qwen-vl-32B-project
bash scripts/setup_environment.sh
python3 scripts/prepare_runtime.py \
  --version optimized \
  --model-path /models/Qwen3-VL-32B-Instruct \
  --gpus 0,1
runtime/optimized/venv/bin/python runtime/optimized/service.py start
runtime/optimized/venv/bin/python runtime/optimized/service.py status
```

启动含加载、编译和 CUDA Graph 预热，实测一次切换加 smoke 约 183 秒，不是固定启动 SLA。`prepare_runtime.py` 不启动服务，不覆盖已有 runtime，自动生成 `0600` 的 `api.key` 且不打印密钥，也可从 `MODEL_API_KEY` 环境变量读取自备密钥。不同版本默认都用 30000 端口，切换时先停止已有实例，或配置独立 GPU/端口。

此仓库的版本配置和源码来自真实运行环境，打包后的准备脚本另外做了路径参数化验证。依赖安装流程依赖可用的 Linux wheels；本次未在全新机器上重新下载并安装全部依赖。cuDNN 9.16.0.29 是显式兼容覆盖，torch 元数据仍声明较旧版本，完整锁定安装使用 `--no-deps` 保留实测组合，不应未经复测交给依赖解析器重选。不要把预期的 cuDNN 元数据冲突当成所有依赖问题都可忽略。

## 步骤 0：保留原始部署

最初版本是单卡 TP1，未开启 DP Attention，使用未修改的 SGLang 相关代码。版本目录保存启动器、配置和后续发生修改的两个源文件；其余代码由固定版本的 SGLang 包提供。它是历史起点，**最终收益表的优化前基线是后续已修复的双卡 DP2，不能把单卡/双卡差异混入优化百分比**。

```bash
python3 scripts/prepare_runtime.py --version original --model-path /models/Qwen3-VL-32B-Instruct --gpus 0 --port 30100
```

## 步骤 1：修复 MRoPE padding

Qwen3-VL 的 `mrope_positions` 是 `[3, token数]`。原逻辑复用普通 batch padding，把第一维按 batch size 填充。batch 小于 3 会产生负长度问题，其他情况下也可能破坏三坐标轴或 token 对齐。

将补齐方向改为最后一维，目标为 `num_tokens`，保留 3 个坐标轴。实际调用 `ForwardBatch._pad_inputs_to_size()` 做五个回归场景：原始失败、prefill 补齐、decode、idle rank、多个请求。详见 [补丁](patches/dp-attention.patch)。

## 步骤 2：隔离视觉编码器的通信范围

不同 DP rank 收到的图片数或模态可能不同，但原视觉编码器仍在共享 TP 组上发 collective，可能与另一 rank 的语言模型通信错位。

DP Attention 开启时，每个 rank 复制完整视觉权重，图片/视频在本 rank 运行，绕过跨 DP rank 的视觉分片 gather 路径。语言模型 FFN 继续用 TP。关闭 DP Attention 的原视觉分片路径保持不变。不能靠“禁止 custom all-reduce”解决这个通信参与集合不一致的问题。

```bash
python3 scripts/prepare_runtime.py --version dp-attention-fixed --model-path /models/Qwen3-VL-32B-Instruct --gpus 0,1
MODEL_RUNTIME=runtime/dp-attention-fixed bash scripts/run_regressions.sh
runtime/dp-attention-fixed/venv/bin/python runtime/dp-attention-fixed/service.py start
MODEL_RUNTIME=runtime/dp-attention-fixed MODEL_API_URL=http://127.0.0.1:30000 \
  runtime/dp-attention-fixed/venv/bin/python tests/test_dp_integration.py
```

验证覆盖 10 个源代码回归、19 个混合图文请求、30,021 token 文本、18,125 token 图文及红蓝变化视频。原始上游代码的 padding 五项失败、视觉三项 DP 用例失败，补丁后全部通过。完整根因与定位链路见 [兼容性文档](docs/dp-attention-compatibility.md)。

## 步骤 3：建立性能基线与 profiler 证据

分别测短文本约 550 token、长文本约 4050 token、1024 方图约 1075 token、8 并发短文本。每项预热，固定输出 128 token，随机 nonce 尽量避免长前缀命中。

- TTFT：请求发送至首个非空流式文本 chunk。
- 本轮 TPOT：`(HTTP 完成时间 - 首个非空 chunk 时间)/(completion_tokens - 1)`。不使用“最后可见字符”的时间低估不可见 token。
- 吞吐：输出 token 总数除以批次总耗时，包含 prefill、排队和流式开销。
- torch.profiler：CPU + CUDA，record_shapes=true，32 个输出 token；按 CUDA Graph correlation 归属 kernel，去掉前 5 和后 2 步。
- 运行空泡：GPU kernel/memcpy/memset 时间并集以外的区间，不能等同于 SM 空闲或带宽利用率。

原 DP Attention decode 的 GEMM 类占约 72%–74%，通信约 5%–10%，每步约 1620 个 kernel。短 prefill 有明显 CPU dispatch/同步间隙。前处理把展开后的 24 MiB 图像 patch 输入回传主机，scheduler 又上传各 GPU。[原分析](docs/profiling-report.md)

## 步骤 4：普通 TP2 取代 DP Attention

统一全局请求上限 8、chunked prefill 4096、graph max batch 8，比较双卡布局。普通 TP2 减少 DP 布局交换和 idle rank attention 的冗余工作：短文本 TPOT 约 15.07 降到 12.35 ms/token，8 并发吞吐约 447.0 提升到 521.0 token/s。

这属于具体模型和负载的选择，不推导“DP Attention 对所有模型都慢”。DP 修复版继续保留，适用于后续更高并发/其他负载的对照。

## 步骤 5：torch.compile 与 CUDA Graph

打开 `--enable-torch-compile --torch-compile-max-bs 8`，预热 batch 1/2/4/8。普通 TP2 基础上，decode 约 12.35 降到 11.88 ms/token，吞吐约 521.0 提升到 569.2 token/s。

服务原先已经使用 CUDA Graph。本步骤增加编译与图内算子优化，不能把所有收益描述为“首次开启 CUDA Graph”。新 shape 的编译成本应放在预热阶段。

## 步骤 6：FlashInfer 通信融合

单独测试 `--enable-flashinfer-allreduce-fusion`，再测试与 compile 的组合。单独开启只有小幅收益；组合后 decode 约 11.29 ms/token，8 并发吞吐约 570.7 token/s，与 compile 单独开启接近。

trace 每步 kernel 从普通 TP2 的 1353 降到通信融合的 1289，再降到 compile + 融合的 1158。数字来自真实 kernel trace，不把 kernel 数下降直接换算为同幅延迟下降。

## 步骤 7：多模态 CUDA IPC 消除主机往返

先测 `--keep-mm-feature-on-device`，没有稳定图像收益。再显式开启：

```text
--keep-mm-feature-on-device
SGLANG_USE_CUDA_IPC_TRANSPORT=1
SGLANG_MM_FEATURE_CACHE_MB=1024
```

单图 TTFT 从约 252 ms 降到 109 ms。scheduler trace 中每 rank 的 24 MiB H→D 消失，另一张 GPU 使用 24 MiB P2P 复制。仍存在设备内拷贝，不能称为全路径零拷贝。缓存/传输池多占用约 1.5 GiB，KV token 容量从 422,433 降至 415,005。

图文 19 项、质量 16 项、72 张独立 1024 方图的 8 并发、长上下文与视频通过，用于检查串图、生命周期和复用。保留 Radix Cache，相同长前缀短答案完整请求从约 281 ms 降至约 93 ms；这个结果不是流式 TTFT，也不证明 ViT 被自动缓存。

## 步骤 8：实现并筛选进一步融合

| 方案 | 实际实现 | 实测结论 |
|---|---|---|
| Piecewise CUDA Graph | 修复空 `sym_shape_indices` 的静态分区回退 | 短文本 TTFT 76.4 ms、单图 322.3 ms，回退，未上线 |
| Q/K RMSNorm + MRoPE | 新 Triton kernel，token 数从 constexpr 改为运行时参数 | 单 token 微基准 9.55 μs 降到 1.71 μs，整模型 decode 11.79 ms/token |
| FP8 | 修复视觉 MLP per-channel scale `[1,N]` 与 `[N,1]` 布局不匹配 | decode 9.44 ms/token，吞吐 680.7 token/s，但未完成业务精度评估 |

QK 融合最大绝对差 0.03125，最少逐元素完全相等比例 78.8%，通过宽松的 BF16 容差，不是逐位等价。它没有融合 KV cache 写入。FP8 通过 35 项小规模检查，但未叠加 compile/IPC，单图 TTFT 332.1 ms。两者保留为实验方案，未默认改变线上精度或替换成熟算子。[实验说明](experiments/README.md)

## 步骤 9：组合验证、切换与回滚

最终 BF16 组合在隔离 GPU 上先验证，再在原 GPU 0/1 切换并各测 5 轮。验证文本、图像、多图、OCR、视频、32K 范围内长输入和鉴权/流式。临时实例退出，最终服务保持原端口、模型 ID、API key、32K 上下文和最大并发 8。

```bash
# 原版本已经运行时，先停止，避免占用相同 GPU 和端口。
runtime/dp-attention-fixed/venv/bin/python runtime/dp-attention-fixed/service.py stop
runtime/optimized/venv/bin/python runtime/optimized/service.py start
MODEL_RUNTIME=runtime/optimized MODEL_API_URL=http://127.0.0.1:30000 \
  runtime/optimized/venv/bin/python tests/smoke_test.py

# 回滚到保留的 DP 修复版
runtime/optimized/venv/bin/python runtime/optimized/service.py stop
runtime/dp-attention-fixed/venv/bin/python runtime/dp-attention-fixed/service.py start
```

不同 runtime 默认各有独立密钥。如果业务切换需要保持同一密钥，准备两个版本时从同一个安全环境变量提供 `MODEL_API_KEY`。源启动器来自 Linux `/proc` 环境，不能在 macOS 直接作为推理守护进程使用。

## 复现基准

```bash
MODEL_RUNTIME=runtime/optimized runtime/optimized/venv/bin/python \
  benchmarks/opt_probe.py --port 30000 --label rerun --rounds 5
MODEL_RUNTIME=runtime/optimized runtime/optimized/venv/bin/python \
  benchmarks/opt_probe.py --port 30000 --label rerun --profile --cases short,image
MODEL_RUNTIME=runtime/optimized runtime/optimized/venv/bin/python \
  benchmarks/quality_compare.py 30000 rerun
```

视频/IPC 生命周期脚本需额外提供本地红色变蓝色视频 `MODEL_TEST_VIDEO`。profiling 与无 profiler 性能测试分开执行。脚本不会打印鉴权密钥，原框架启动日志可能包含启动参数，日志不要提交。

## 已知边界与下一步

本轮没有实现 ViT 专项融合、跨进程统一 request span、prefill/decode 分离、KV FP8 或投机解码，也没有通过降低图片分辨率或提高并发上限包装收益。下一个实验应使用真实业务数据验证 FP8/自定义 QK 的质量，再单独测它们与当前优化组合的交互。硬件计数器需要额外用 Nsight Compute 采样热点，现有 profiler 不能直接证明 HBM 饱和。

### PPT 参考资料更新（2026-09-22）

最新版含四次课实验、验证结果，以及第 102 页 GitHub 参考代码、第 103 页知乎延伸阅读。知乎 4 篇文章已于 2026-09-22 在浏览器逐个打开并核对标题、作者和正文；PPT 标明了各文章与项目的关系及适用范围。旧版演示文稿可通过 Git 历史查看。
