# 工程打包与复现说明

## 保存了什么

`versions/` 保存原始单卡、DP 修复版、最终优化版的配置、启动器与发生过修改的两个 SGLang 完整源文件。其余框架代码来自固定版本 `sglang==0.5.6.post2`，避免复制整个虚拟环境。权重 revision、依赖快照和源文件 SHA256 一并记录。

历史文件中的模型路径、内网 IP、服务器别名已替换。源码逻辑没有为发布重新设计。`scripts/prepare_runtime.py` 另外提供便携的部署目录准备功能，复制固定版本 SGLang 包，再叠加所选版本的完整源文件，链接虚拟环境并生成运行配置。

准备脚本做了两个打包层适配：所有版本都显式加载自己的 overlay，包括原始版本；`status` 请求使用配置中的端口，而不是历史代码写死的 30000。`versions/*/service.py` 保留原快照逻辑，适配仅发生在生成的 runtime 中。

## 这次额外检查

- 在原远程 Linux 环境，用现有固定版本虚拟环境实际生成了独立 DP 修复 runtime。
- 从这个打包 runtime 的 overlay 导入 SGLang，十个源码回归全部通过。
- 将最小 DP 补丁应用到原始快照后，两个文件与修复快照逐字节一致。
- 本次只准备目录和运行无权重回归，没有启动第二个模型服务，也没有修改当前服务配置。
- 依赖安装脚本未在全新机器重新执行下载和完整安装；需留意特定版本 wheel 与 cuDNN 元数据覆盖说明。

## 测试数据和 trace 的脱敏

`benchmarks/results/` 保留逐请求结果。`evidence/` 包含此前报告、脚本和 CPU/CUDA trace。归档脚本中的 `/deployment/qwen3-vl-32b` 等是脱敏占位路径；复现请用仓库顶层的 `scripts/` 与 `benchmarks/` 可配置入口。

脱敏保留事件时间、耗时、shape、kernel 名称和请求 token 统计。压缩 trace 的字节已改变，因此各 `trace-index.md` 已重新计算 SHA256，不再沿用原 gzip 的哈希。目录中不会包含 `api.key` 或原始服务日志。

框架启动日志可能含 API key 参数。`.gitignore` 排除日志、runtime、缓存、权重和密钥文件。新运行产生的日志仍应按凭据文件管理。
