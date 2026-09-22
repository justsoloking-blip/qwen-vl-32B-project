# 第二课：DP Attention 两个兼容性修复：参考答案

MRoPE位置矩阵的行数3是t/h/w坐标轴，列数才是token数。[3,78]补到128，应在右边增加50列，得到[3,128]：

```python
mrope_positions = torch.nn.functional.pad(
    mrope_positions, (0, num_tokens - mrope_positions.shape[1])
)
```

原逻辑把矩阵第一维补到batch size。batch=1时会申请负数长度，batch≥3时也可能得到错误行数/列数。一个重要反例是batch恰好3且无需增加token时原代码可能碰巧通过，因此“一个请求通过”不能证明补丁正确。

视觉问题与张量补齐是不同故障：不同DP rank处理不同模态/图片数，却在共享TP组调用不匹配的collective序列。修复时，让DP Attention模式复制完整视觉权重，并在本rank执行视觉编码器，语言FFN继续TP。分支条件是 `self.use_data_parallel and not is_dp_attention_enabled()` 才走原视觉分片路径。

参考补丁：common/project/patches/dp-attention.patch。CPU验证从冻结源码AST提取真实方法，不重写方法体；视觉计算和collective用mock替代，因此只证明张量/dispatch契约，没有执行GPU通信或SGLang完整初始化。

## 执行与证据

在包根目录运行：

```bash
.venv-local/bin/python 02_dp_fix/verify.py
```

本次执行14个padding检查、8个image/video dispatch检查、4个构造参数检查，全部符合预期。原始padding在7个场景中6个违反目标不变量，另1个为预期的碰巧通过；修复后7个均通过。完整日志在results/。
GPU运行入口：`python3 gpu_lab.py --stage fix --model-path /models/Qwen3-VL-32B-Instruct`，会另跑完整SGLang回归和图文请求。

本地CPU/归档检查与新GPU验证分别记录，详见下节。

## 新增 GPU 实测与验收

已在独立Linux环境导入完整SGLang源码执行回归：原始padding五例全部失败（3失败、2错误），原始视觉dispatch五例中3例失败、2例通过；修复后两组共10项通过。原始失败日志见verification/gpu/upstream-regression，修复日志见verification/gpu/candidates/A_dp2/logs/A_dp2-regressions.log。双卡A_dp2混合图文19请求通过，另有16项质量检查通过。源码单元测试的视觉mock与真实双卡集成分开记录，不能将前者误称为collective实测。
