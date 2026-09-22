# 第四课：手写融合算子与采用决策：参考答案

源码来自common/project/experiments/fused_qk_mrope.py，是项目手写实验内核。Triton提供DSL与编译工具。grid=(N,NQ+NK)，TP2下NQ=32、NK=4，即N×36个program。一个program处理一个token的一个Q头或K头，program不等于一个CUDA线程。

每个头128维拆成x0=x[:64]、x1=x[64:]。r=rsqrt(sum(x0²+x1²)/128+eps)，a=BF16(x0*r*w0)，b=BF16(x1*r*w1)，然后y0=a*cos-b*sin，y1=b*cos+a*sin。频率通道按[0,1,2]重复20次再加4个0选择t/h/w，共[24,20,20]。Norm后的BF16舍入是参考路径的一部分。

输入Q/K可为QKV split产生的非连续视图，所以token寻址使用输入stride。输出新分配连续Q/K。融合范围不包含QKV矩阵乘、Attention点积/softmax或KV cache写入。

N若作为constexpr，不同token数可能产生不同特化和编译开销。源码已改为运行时N，其他元参数仍可能导致新变体。必须区分首次编译、首次新长度和重复长度测量，不能把首次成本当作稳态性能。

历史GPU微基准TP2单token约9.55→1.71μs，局部约5.58倍，但整模型decode仅12.35→11.79ms/token，因为GEMM、通信等仍占大量时间。历史最大绝对误差0.03125，不能称逐位一致。主服务未部署此kernel。

本次CPU测试分别使用两种表达方式计算Norm与旋转，16组通过既定容差；它只验证公式/布局，不验证Triton编译、CUDA执行或FlashInfer reference的一致性。

## 执行与证据

在包根目录运行：

```bash
.venv-local/bin/python 04_fusion/verify.py
```

CPU结果：results/verification.json；历史微基准重新计算：results/historical_microbench_recomputed.json。
GPU微基准及服务：`python3 gpu_lab.py --stage fusion --model-path /models/Qwen3-VL-32B-Instruct`。
形状特化实验：在Linux GPU项目环境运行 `common/project/.venv/bin/python 04_fusion/compare_specialization_gpu.py`。这两条路径现已在Linux GPU执行，结果见下节。
采用答案：当前默认BF16组合；融合kernel、Piecewise和FP8保留实验。优化未变快或质量有差异时，如实保留结果，不以强行上线作为作业通过标准。

本地CPU/归档检查与新GPU验证分别记录，详见下节。

## 新增 GPU 实测与验收

实际CUDA/FlashInfer参考对照16组通过，最大绝对误差0.03125。TP2单token参考9.532μs、融合1.701μs；本次整模型B为12.3505 ms/token，Q为11.7803 ms/token。Q的混合并发出现1300返回130的反例，小型质量16/16，课程题30/40。该候选被拒绝采用，失败日志见verification/gpu/rejected-qk。

原始微基准：verification/gpu/candidates/Q_qk/optimization/fused-kernel-test.json。形状特化12次实测：verification/gpu/specialization-gpu.json。runtime N对新552/553长度没有出现constexpr版本约90ms的开销；首次调用混有初始化，重复调用计时也包含Python和同步。

H默认组合不包含自定义QK内核。课程题严格匹配39/40，一题返回5 8 4 2而非5842；忽略OCR数字间空格时内容40/40。这是对失败类型的事后分类，不修改原评分或将其标为严格通过。本次拒绝部署Q：除了BF16非逐位一致，还观察到实际混合并发答案错误。性能提升与容差通过均不足以越过该质量失败。回滚步骤：停止当前实验runtime，再启动已保存的A_dp2 runtime，等待/health成功，运行tests/smoke_test.py，最后停止实验。此步骤已实际执行，证据verification/gpu/rollback.json。

新增positions非连续边界：实际执行check_position_stride_gpu.py，将[3,17]缓冲区切成N=1/4/8后与同值连续张量比较，最大绝对差分别10.3125、14.25、12.875。实验内核POS+axis*N+t遗漏了positions.stride(0)，应根据真实行列stride寻址。SGLang CUDA graph确有buffers.mrope_positions[:, :num_tokens]切片。这证实内核存在布局缺陷，但未证明所有服务错误只有这一根因。原16组微基准只有连续positions，覆盖不充分。见verification/gpu/position-stride-gpu.json。

