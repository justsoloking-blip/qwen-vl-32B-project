# 本次 GPU 复现的方法与修复记录

本次从一台没有项目与 SGLang 环境的 Linux 服务器开始，创建独立项目目录与 venv，下载固定 revision 的完整 Qwen3-VL-32B-Instruct。使用 H200 的 GPU 0、1，NVLink 互联。服务器 GPU 6 原有其他任务，本次未操作它；因此可以说测试 GPU 对独立，不能说整台主机独占。

环境与全量权重校验在 `verification/gpu/environment/`。原始请求、配置、质量结果、trace 在 `verification/gpu/candidates/`。不包含权重、venv、API key 或私有服务启动日志。

## 真实运行中修正的三个实验脚本问题

1. **模型列表可用不代表模型就绪。** 最初以 `/v1/models` 的 HTTP 200 判断就绪，随后 `/health` 仍为 503，smoke 失败。现在 `gpu_lab.py` 轮询 `/health`，成功后才执行请求。局部回归覆盖 503→200 和超时，结果见 `verification/readiness-test.json`；随后真实 DP2 smoke 通过。
2. **端口释放后的 TIME_WAIT。** 第一组完成并停止服务后，下一阶段的端口预检查报 `Address already in use`，但没有监听进程。预检查 socket 增加 `SO_REUSEADDR`，仍然拒绝正在监听的端口。复现入口无需换端口或杀其他任务。
3. **以减号开头的随机密钥。** 一次运行生成了以 `-` 开头的 key，`--api-key VALUE` 触发 argparse 的 `expected one argument`。改为 `--api-key=VALUE`，不更改密钥内容。runner 在新建隔离 runtime 后修正其启动器；冻结的原项目快照保持原样。真实运行中已在当前 runtime 与远端后续生成模板应用等价修复。局部解析与端口回归见 `verification/runner-regressions.json`。

这些是教学复现工具的修复，不是模型性能收益来源，也不是 DP Attention 的两个模型兼容性补丁。

## 运行顺序

`fix` 阶段的 A_dp2 同时满足第一课的部署基线，因此无需重复跑 baseline。后续使用 `--labels` 跳过已经完成的配置。每个配置均使用独立 runtime、相同 GPU 对和端口，结束后停止本实验服务。

```bash
python3 gpu_lab.py --stage fix --model-path /models/Qwen3-VL-32B-Instruct --rounds 5
python3 gpu_lab.py --stage ablation --labels B_tp2,C_compile,D_fusion,E_compile_fusion,F_keep_device,G_ipc --model-path /models/Qwen3-VL-32B-Instruct --rounds 5
python3 gpu_lab.py --stage fusion --labels Q_qk --model-path /models/Qwen3-VL-32B-Instruct --rounds 5
python3 gpu_lab.py --stage final --model-path /models/Qwen3-VL-32B-Instruct --rounds 5
```

B_tp2 是 Q_qk 的整模型对照。Q_qk 为实验内核，H_final 不包含该内核。原始源码失败测试、形状特化实验另行执行，详见各课答案。

## 数据与结论边界

- 各配置使用同一请求生成器、输入尺寸、并发与输出长度；请求包含随机盐，输入 token 数略有不同，原始 usage 中有记录。不能称请求逐字相同。
- 每类 5 轮，单配置 20 批、55 条计时请求。预热请求与 profiler 请求另计。顺序运行没有随机化配置顺序，不排除时间漂移；小幅差异应追加交错复测。
- 吞吐按总输出 token / 总批次墙钟时间计算。报告中 TPOT 是客户端观察值，包含请求协议耗时；不等同于单 kernel 时间。
- Profiler 用 32 输出 token，基准用 128 输出 token，两者分开。两个 GPU 的 kernel 时长之和不能直接当请求延迟。
- 40 条合成题只检查整数、OCR、颜色与多图顺序，不覆盖视频或业务质量。容差通过也不等于 BF16 逐位相同。
- `run_verified_local.py` 只运行 CPU 与归档重算；`verify_fresh_results.py` 只核验已经实际采集的 GPU 证据。两者都不会在本地重新执行模型。学生需要按 GPU 命令重新运行才能获得自己的实测结果。

最终通过状态由 `verification/full-course-verification.json` 与各项原始证据共同确定。

## 自定义 QK 候选的真实质量失败

首次 Q_qk 混合并发测试要求原样输出 `1300`，实际输出 `130`，触发精确匹配断言。原始失败日志保存在 `verification/gpu/rejected-qk/first-integration-failure.log`。此前A同一测试通过。

为完成该候选的实验分析，runner对Q的质量检查保留原断言和非零返回码，继续采集后续质量、性能和内核数据，在阶段清单中记录 `quality_failures`。其他配置仍遇错停止。`execution_completed`只表示实验执行完，不表示质量通过。即使后续重跑偶然通过，也保留首次反例并拒绝部署Q。

本次没有定位到该输出差异的具体根因，因此不能断言一定是某个舍入步骤导致。该失败已经足够阻止采用；如要继续开发，需要固定并发调度、保存逐token logits并比较原生/融合路径，再定位差异。

H最终组合的课程题严格评分为39/40，唯一失败是OCR将`5842`输出为`5 8 4 2`。事后按“只忽略OCR数字间空白”分类，数字内容为40/40。保留原始严格失败和非零返回码；runner只为完成后续性能测量而记录该失败并继续，不能据执行完成宣称严格格式通过。对有严格输出格式要求的应用，该问题仍需处理。

追加边界实验已确认Q内核的positions行stride缺陷，结果见verification/gpu/position-stride-gpu.json；完整微基准的16组仅覆盖连续positions。该边界诊断在空闲GPU2执行，不使用其结果计算GPU0/1的服务性能。
