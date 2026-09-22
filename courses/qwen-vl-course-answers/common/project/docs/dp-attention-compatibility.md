# Qwen3-VL-32B 的 DP Attention 兼容性问题

## 适用范围

本次修复针对 SGLang 0.5.6.post2、Qwen3-VL-32B-Instruct、两张 H200、TP=2 / DP=2、BF16、32K 上下文。模型是稠密 Transformer，非 MoE、非 GDN。源码使用 `Qwen3VLMoeVisionModel` 作为共享视觉实现类名，不代表这个 32B 模型本身是 MoE。

修复修改两个文件：

- `sglang/srt/model_executor/forward_batch_info.py`
- `sglang/srt/models/qwen3_vl.py`

[最小补丁](../patches/dp-attention.patch) · [原始源码](../versions/original/src/sglang/) · [修复版源码](../versions/dp-attention-fixed/src/sglang/)

## 问题一：MRoPE 坐标轴被当成 batch 维补齐

### 现象与原因

打开 DP Attention 后，prefill、decode 或 idle rank 的 batch padding 会处理多模态位置张量。普通 `positions` 是一维 token 位置，而 `mrope_positions` 实际为 `[3, N]`，三个轴对应多模态坐标。原代码：

```python
self.mrope_positions = self._pad_tensor_to_size(self.mrope_positions, bs)
```

通用方法沿第 0 维处理，同时目标使用 batch size。它既把三个坐标轴误当成请求数，也把 token 数误当成 batch size。原始单请求场景会遇到负 padding 长度，其他形状可能破坏三轴和 token 数的一致性。

### 修改

```python
self.mrope_positions = torch.nn.functional.pad(
    self.mrope_positions,
    (0, num_tokens - self.mrope_positions.shape[1]),
)
```

右侧补齐最后一维，结果保持 `[3, num_tokens]`。原有 token 的三个坐标全部保留，新增位置填零。该调用点的调度契约要求目标 token 数不小于当前数，补丁没有引入新的裁剪行为。

### 验证

直接调用真实 `ForwardBatch._pad_inputs_to_size()`，无需加载 32B 权重。

| 用例 | 输入 `[3,N]` | 目标 token / batch | 验证点 |
|---|---|---|---|
| 原始单请求 prefill | `[3,78]` | 78 / 1 | 原错误场景 |
| prefill 补齐 | `[3,5]` | 8 / 2 | token 轴扩展 |
| decode | `[3,1]` | 4 / 4 | batch 与 token padding |
| idle rank | `[3,0]` | 4 / 4 | 零 token 参与图捕获 |
| 多请求 | `[3,7]` | 10 / 3 | 保持三坐标轴 |

同时校验 shape、原值、新区域为零、dtype。原代码五项失败，修复后五项通过。[测试源码](../tests/test_dp_mrope_padding.py)

## 问题二：视觉 collective 与语言模型通信错位

### 现象与定位

DP Attention 允许各 rank 收到不同请求。某 rank 可能处理图片，另一 rank 可能只处理文本或为空。原 Qwen3-VL 视觉模型仍使用整个 TP 通信组分片，图片/视频特征路径还可能进入跨 rank 的视觉 sharding helper。

这样的 collective 要求参与者执行相同的通信序列，但另一 rank 此时可能已进入语言模型 FFN 的 TP 通信。实际表现为请求卡住或通信调用不匹配。对照两 rank 调用栈、张量形状和视觉 dispatch 后定位到参与集合不一致。仅禁止 custom all-reduce 不能改变这个逻辑条件，因此不是有效修复。

### 修改一：DP Attention 下复制视觉权重

```python
self.use_data_parallel = (
    get_global_server_args().mm_enable_dp_encoder
    or is_dp_attention_enabled()
)
```

这使视觉模型在该模式下使用完整的本地视觉编码器。每个 DP rank 独立计算自己请求的图像或视频。代价是每个 rank 保留完整视觉权重，应计入显存预算。

### 修改二：图片和视频绕过跨 rank 视觉 sharding helper

两个模态入口均采用：

```python
if self.use_data_parallel and not is_dp_attention_enabled():
    return run_dp_sharded_mrope_vision_model(...)
else:
    return self.visual(...)
```

语言模型稠密 FFN 仍使用 TP。补丁不是“所有通信都取消”，而是避免各 rank 请求不对称时视觉编码器进入共享的 TP collective 序列。原来没有 DP Attention 的视觉分片分支保留。

### 验证

| 源码回归 | 修复后 |
|---|---|
| DP Attention 开启时视觉权重复制 | 通过 |
| 图片使用本地视觉 encoder | 通过 |
| 视频使用本地视觉 encoder | 通过 |
| 非 DP Attention 图片 sharding 保留 | 通过 |
| 非 DP Attention 视频 sharding 保留 | 通过 |

前三项在原代码失败，后两项确保没有破坏原功能。[测试源码](../tests/test_dp_vision_dispatch.py)

## 验证链路与复现

1. 在上游原文件运行两个回归文件，确认失败模式。
2. 对同一 SGLang 包的独立 overlay 应用最小补丁。
3. 重跑十个源码回归，确认通过。
4. 启动 TP2/DP2 服务，检查鉴权、健康、模型名和流式输出。
5. 运行 19 个混合请求，覆盖多图、不同尺寸、不等长文本、不同模态和最多 8 并发。
6. 验证 30,021 token 文本、18,125 token 图文和 4 秒红蓝变化视频。

```bash
python3 scripts/prepare_runtime.py --version dp-attention-fixed \
  --model-path /models/Qwen3-VL-32B-Instruct --gpus 0,1
MODEL_RUNTIME=runtime/dp-attention-fixed bash scripts/run_regressions.sh
runtime/dp-attention-fixed/venv/bin/python runtime/dp-attention-fixed/service.py start
MODEL_RUNTIME=runtime/dp-attention-fixed MODEL_API_URL=http://127.0.0.1:30000 \
  runtime/dp-attention-fixed/venv/bin/python tests/test_dp_integration.py
MODEL_RUNTIME=runtime/dp-attention-fixed MODEL_API_URL=http://127.0.0.1:30000 \
  runtime/dp-attention-fixed/venv/bin/python tests/test_dp_long_context.py
```

如需手工验证补丁，在含原始 `sglang/` 的 overlay 根目录运行：

```bash
patch --dry-run -p1 < /path/to/patches/dp-attention.patch
patch -p1 < /path/to/patches/dp-attention.patch
```

仓库快照已包含修复，不应对修复版重复应用补丁。服务必须在修改源码后重新启动，不能依靠修改运行中进程的磁盘文件热替换。

## 为什么最终优化版关闭 DP Attention

修复证明该组合能够正确运行。性能测试另行显示，在本次短文本、4K 文本、单图和 8 并发负载下，普通 TP2 减少布局交换与 idle rank 冗余，效果更好。最终配置因此选普通 TP2，仍保留完整 DP 修复版本与回滚路径。

当前补丁没有声称支持所有 TP/DP 组合、跨节点通信或未来版本。升级 SGLang、修改视觉分片策略和扩大上下文时，都应先跑回归，再做混合请求压力验证。功能检查通过也不能替代长期稳定性和真实业务负载测试。
