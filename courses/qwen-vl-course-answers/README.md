# 四次实验作业与参考答案

每课一个文件夹，含 ASSIGNMENT.md、ANSWER.md、verify.py 和 results/。

**验证状态：四课本地检查通过，完整 Linux H200 双卡 GPU 实验已执行并保留失败结果。** 默认组合H通过基础质量检查，课程题严格39/40（1题OCR空格格式差异）；实验Q出现混合并发答案错误，已拒绝采用。 固定版本完整权重校验、九配置基准、DP兼容性、Profiler与Triton实验均有原始证据。总报告见[FRESH_GPU_REPORT.md](FRESH_GPU_REPORT.md)，机器可读验收见verification/full-course-verification.json。

|文件夹|本次实际验证|
|---|---|
|01_baseline|完整模型、环境、接口、4类×5轮基线、40条课程数据|
|02_dp_fix|原始源码失败、修复10项回归、双卡19混合请求|
|03_profiling|九配置、495条计时请求、12份新增GPU trace及12份历史trace|
|04_fusion|16组真实CUDA微基准、12次形状特化调用、整模型与质量、回滚|


## 本地可执行验证

Python 3.12，独立虚拟环境执行，勿安装进教学GPU共享环境：

```bash
python3.12 -m venv .venv-local
.venv-local/bin/python -m pip install -r requirements-local.txt
.venv-local/bin/python run_verified_local.py
```

本次环境为macOS ARM64、CPU PyTorch 2.9.1，完整包版本见 verification/local-pip-freeze.txt。这组CPU依赖与Linux GPU环境的完整锁定依赖不同。脚本失败会返回非零状态；不会把GPU跳过项伪装成通过。

## GPU实测入口（已验证）

common/project 是从项目教学所需文件复制的冻结快照，包含SGLang补丁、配置、回归和基准。完整教学讲解见各课文件。先由教师在Linux准备依赖和模型，再在**独占的GPU对**上执行：

```bash
cd common/project
bash scripts/setup_environment.sh
cd ../..
common/project/.venv/bin/python download_model.py --model-dir /models/Qwen3-VL-32B-Instruct
python3 check_snapshot.py /models/Qwen3-VL-32B-Instruct --sha256
python3 gpu_lab.py --stage baseline --model-path /models/Qwen3-VL-32B-Instruct
python3 gpu_lab.py --stage fix --model-path /models/Qwen3-VL-32B-Instruct
python3 gpu_lab.py --stage ablation --model-path /models/Qwen3-VL-32B-Instruct
python3 gpu_lab.py --stage fusion --model-path /models/Qwen3-VL-32B-Instruct
python3 gpu_lab.py --stage final --model-path /models/Qwen3-VL-32B-Instruct
```

`--gpus 0,1 --port 30100 --venv /absolute/venv` 可覆盖默认值。`--plan`只生成配置，不启动GPU服务。本次已执行完整GPU路径，并修正就绪检查、端口复用和减号开头密钥三个实际问题，见REPRODUCTION_NOTES.md。runner顺序创建隔离runtime，等待服务就绪、执行基准并在finally停止自己启动的服务，不切换现有生产服务。不会自动选择可用GPU，使用前由教师分配空闲GPU。

GPU结果写入 gpu_runs/时间戳/；该目录可能含api.key和带启动参数的私有日志，不应直接打包提交。学生提交去掉密钥、私有地址及启动密钥参数后的结果。目录venv为共享依赖的符号链接，overlay和配置独立，不能在作业中随意修改site-packages。

40条课程质量数据位于01_baseline/data；本次已在Q自定义内核服务和H最终组合逐题验证，具体匹配数见报告。H严格39/40，1题为OCR空格格式差异；Q另有混合并发与数字内容失败，不能标为质量通过。视频样例不包含在该40条集合，需按原项目要求另备红转蓝MP4。

## 模型链接

- 官方仓库：https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct
- 固定版本文件：https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/tree/0cfaf48183f594c314753d30a4c4974bc75f3ccb
- 配置：https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/resolve/0cfaf48183f594c314753d30a4c4974bc75f3ccb/config.json
- 权重索引：https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/resolve/0cfaf48183f594c314753d30a4c4974bc75f3ccb/model.safetensors.index.json

本次核验了官方页、固定revision页面和全部26个仓库文件链接，14个权重分片各读取16字节以确认可下载并检查safetensors头长度。小型配置JSON已下载并解析。**已在远端下载完整约66.7GB权重，14分片SHA256通过，九配置均实际加载并推理。** 完整记录在 verification/model-links.json，链接可用性只代表记录时刻。重新执行 `python3 verify_model_links.py` 可刷新。

## 证据与来源

common/archive 的trace与 common/project/benchmarks/results 的GPU数字是历史实测资产；各课results同时包含本地归档重算和标明fresh-gpu的新增实测分析，新增GPU原始证据在verification/gpu。不要把两者的时间混同。源快照哈希见verification/source_sha256.json。项目许可说明见common/project/THIRD_PARTY_NOTICES.md及environment/SGLANG-LICENSE。

可执行 `python3 verify_fresh_results.py` 对已收集GPU证据重新验收并生成指标与trace分析。这是重算入口，不会启动模型。
