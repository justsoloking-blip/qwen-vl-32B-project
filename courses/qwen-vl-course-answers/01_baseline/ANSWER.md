# 第一课：模型、部署和性能基线：参考答案

文本链：tokenizer → embedding → 64层语言模型 → 输出分布 → 逐token生成。图文路径先由processor构造patch和网格信息，ViT与视觉合并产生视觉特征，并通过多模态输入与DeepStack进入语言模型。MRoPE使用时间、高度、宽度坐标，不能当成普通一维position向量。

TTFT是客户端发出请求到首个非空文本chunk的时间。TPOT使用 `(elapsed_s*1000 - ttft_ms)/(completion_tokens-1)`。并发吞吐使用所有请求的输出token之和除以所有测量批次wall time之和，不能把单请求吞吐相加。

归档重算参考：修复后DP2短文本TTFT 71.5585ms、TPOT 15.0330ms/token；最终部署52.9055ms、11.2523ms/token。8并发输出吞吐447.4846→572.0377 token/s左右，以results/metrics.json完整精度为准。比较双方均为双卡，不能把TP1历史起点当成该提升的基线。

40条教学数据是10条整数计算、10条OCR、10条背景色、10条多图顺序。预期答案由构造过程确定。本地验证文件和清单；新增GPU答题结果见下节。

## 执行与证据

在包根目录运行：

```bash
.venv-local/bin/python 01_baseline/verify.py
```

`results/metrics.json`：历史基准重新计算结果。
`data/manifest.json`：40条样本、预期答案及SHA256。
`results/verification.json`：执行时间、作用范围、模型结构检查。
GPU运行入口：包根目录 `python3 gpu_lab.py --stage baseline --model-path /models/Qwen3-VL-32B-Instruct`。

本地CPU/归档检查与新GPU验证分别记录，详见下节。

## 新增 GPU 实测与验收

本次完整模型加载与smoke通过，A_dp2短文本TPOT为14.9459 ms/token，图像TTFT为268.063 ms，8并发吞吐为450.510 token/s。每类5轮、128输出token。14个权重分片逐一SHA256通过。40条课程样本已分别在Q和H模型服务实际答题，Q为30/40，H严格39/40，忽略OCR数字间空格后内容40/40。

原始证据：verification/gpu/environment、verification/gpu/candidates/A_dp2。Q与H的optimization/course-quality.json保留每题输出。
