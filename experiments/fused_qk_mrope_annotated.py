# 中文教学注释版：Q/K RMSNorm + 交错式 MRoPE 融合算子
# 来源：publish/experiments/fused_qk_mrope.py
# 仅添加注释、展开排版，保留原代码的计算逻辑和参数。
#
# 功能：
#   Q → 按 head 做 RMSNorm → MRoPE 旋转 → 输出 Q′
#   K → 按 head 做 RMSNorm → MRoPE 旋转 → 输出 K′
# 两条路径共用一次 kernel launch，但 Q/K 各自独立计算。
# 不包含 QKV 投影、Attention 主计算、V 处理或 KV Cache 写入。
#
# 适用范围：本实验 BF16，head_dim = rotary_dim = 128，NeoX 配对布局，
# MRoPE sections = [24, 20, 20]，且采用本代码固定的交错方式。
#
# 已知问题：POS 地址计算假设 positions 为连续的 [3,N] 张量。
# 非连续切片可能读错。此注释版没有修复这一缺陷，不能作为通用已验收内核。
# 本候选未进入项目最终默认部署方案。

import os
"""Experimental Qwen3-VL BF16 Q/K RMSNorm + interleaved MRoPE.
Narrow contract: head_dim=rotary_dim=128, Neox layout, sections [24,20,20].
Outputs remain separate Q/K tensors; this does not fuse the KV cache write.
"""
import torch
import triton
import triton.language as tl


# triton.jit：由 Triton 编译这个函数，使其在 GPU 上执行。
# Python 包装函数在文件末尾，负责分配输出并启动这个 kernel。
@triton.jit
def _qk_norm_mrope(
    Q, K,       # Q/K 输入张量的 GPU 指针。
    QW, KW,     # Q/K RMSNorm 的缩放权重，各有 128 个数。
    POS,        # positions 的 GPU 指针，逻辑形状 [3,N]。
    CACHE,      # cos/sin 查找表指针；注意：不是 KV Cache。
    OQ, OK,     # 最终 Q′/K′ 输出张量的 GPU 指针。
    N,          # 本轮 token 数量；运行时参数，不是历史上下文总长度。
    NQ: tl.constexpr,  # 本 rank 的 Q head 数。
    NK: tl.constexpr,  # 本 rank 的 K head 数。
    QS: tl.constexpr,  # Q 相邻 token 行的地址跨度，单位是元素，不是字节。
    KS: tl.constexpr,  # K 相邻 token 行的地址跨度。
    EPS: tl.constexpr, # RMSNorm 稳定项，默认由外部传入 1e-6。
):
    # constexpr 表示编译时可知的参数；不同取值可能产生不同编译变体。
    # 它不表示“训练参数”或“所有模型都固定不变的常数”。

    # ① 确定当前 program 负责哪个 token、哪个 head。
    # grid=(N,NQ+NK)，每个 program 处理一个 token 的一个 Q 或 K head。
    # program_id(0/1) 返回当前 program 在该方向上的坐标，从 0 开始。
    # 0/1 是方向编号，不是 GPU 编号，也不是 CUDA 线程编号。
    t = tl.program_id(0)  # token 编号；这里 t 不是 MRoPE 时间坐标。
    h = tl.program_id(1)  # head 编号；这里 h 不是图片高度坐标。
    d = tl.arange(0, 64)  # 向量 [0,1,...,63]，表示 64 个旋转对。

    # ② 根据 h 选择 Q 或 K，再定位该 head 的起始地址。
    # 例如 NQ=32、NK=4：h=0..31 处理 Q；h=32..35 处理 K。
    if h < NQ:
        # Q 输入逻辑形状 [N,NQ*128]。
        # t*QS 跳到 token 行，h*128 跳到该行的 head。
        # 输入使用真实行跨度，输出是新分配的连续张量。
        src = Q + t * QS + h * 128
        w = QW
        dst = OQ + t * NQ * 128 + h * 128
    else:
        # K 的 head 编号从 0 开始，所以减去 NQ。
        # K 指针本身已指向 K 视图，不需要再跨过 Q 的区域。
        src = K + t * KS + (h - NQ) * 128
        w = KW
        dst = OK + t * NK * 128 + (h - NQ) * 128

    # ③ 读取一个 head 的 128 个数，拆成前后两半。
    # NeoX 旋转配对是 (0,64)、(1,65)、...、(63,127)。
    # x0、x1 各有 64 个元素；转换到 FP32 计算 RMSNorm 统计量。
    x0 = tl.load(src + d).to(tl.float32)
    x1 = tl.load(src + d + 64).to(tl.float32)

    # ④ 求当前 head 的均方根倒数。
    # x0*x0+x1*x1：把两个半区对应元素的平方相加，得到 64 个数。
    # tl.sum(...,axis=0)：把这 64 个数求和，相当于累加全部 128 维的平方。
    # 除以 128 得均方；加 EPS 后取 rsqrt，即 1/sqrt(...)。
    # 不跨 token/head 归约，也不减去均值。
    # EPS 防止全零输入导致除零；本例 1/sqrt(0+1e-6)=1000。
    inv = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1, axis=0) / 128 + EPS)

    # Norm 权重由原模型提供，每个维度一个权重。
    # 同一套 QW 在所有 Q heads/token 上复用，KW 在所有 K heads/token 上复用。
    w0 = tl.load(w + d).to(tl.float32)
    w1 = tl.load(w + d + 64).to(tl.float32)

    # ⑤ 完成 RMSNorm：归一化后乘对应的缩放权重。
    # Preserve the BF16 intermediate rounding of the original RMSNorm.
    # 原路径 Norm 结果为 BF16，这里显式转回输入元素类型以保留该舍入步骤。
    # 本实验 Q/K 均为 BF16；不能据此保证最终与参考实现逐位完全一致。
    a = (x0 * inv * w0).to(Q.dtype.element_ty)
    b = (x1 * inv * w1).to(Q.dtype.element_ty)

    # ⑥ 为每个旋转对选择 MRoPE 的坐标轴。
    # ax=0 表示时间 T，ax=1 表示高度 H，ax=2 表示宽度 W。
    # d=0..59 时按 T,H,W 交错 20 次；d=60..63 全部使用 T。
    # 总计 24 对 T、20 对 H、20 对 W。
    # ax 与 d 一样是长度 64 的向量，并非整个 head 只选一个轴。
    ax = tl.where(
        (d % 3 == 1) & (d <= 60), 1,
        tl.where((d % 3 == 2) & (d <= 60), 2, 0),
    )

    # ⑦ 从位置表读取当前 token 在所选轴上的位置。
    # 连续 positions 的逻辑布局：
    #   第 0 行：所有 token 的时间坐标，共 N 个元素
    #   第 1 行：所有 token 的高度坐标，共 N 个元素
    #   第 2 行：所有 token 的宽度坐标，共 N 个元素
    # POS+ax*N+t 相当于 positions[ax,t]，tl.load 才真正读取数据。
    # 例如 N=4、ax=1、t=2：读第 3 个 token 的高度坐标，偏移为 6。
    # POS 是指针；pos 是读出的整数位置向量，不是角度。
    #
    # 已知 stride 缺陷：若 positions=buffer[:,:4]，原 buffer 形状为 [3,17]，
    # positions 的 shape 为 [3,4]，但行跨度仍可能是 17。
    # 此时应按真实 stride 寻址，而本代码仍用 N=4，可能读错行。
    # 教学版原样保留这行，避免把“注释版”误认为“修复版”。
    pos = tl.load(POS + ax * N + t)

    # ⑧ 查出位置旋转用到的 cos/sin。
    # CACHE 假设每行连续存放 128 个元素：前 64 个 cos，后 64 个 sin。
    # pos 选择位置行，d 选择旋转对对应的频率。
    # 同样的 pos，若 d 不同，频率和旋转角通常也不同。
    # 三角函数已预计算好，这里只读取缓存，不现场调用 sin/cos。
    c = tl.load(CACHE + pos * 128 + d)
    s = tl.load(CACHE + pos * 128 + d + 64)

    # ⑨ 旋转每对 (a,b)，直接写出最终 Q′ 或 K′。
    # a′=a*cos-b*sin；b′=b*cos+a*sin。
    # Norm 中间结果不必先写出独立全局内存张量，再由另一个 kernel 读回。
    tl.store(dst + d, a * c - b * s)
    tl.store(dst + d + 64, b * c + a * s)


# Python 包装函数：在主机端组织输入、分配输出、启动 GPU kernel。
# 输入约定：
#   q: [N,NQ*128]，k: [N,NK*128]，head 内各维连续；Q/K 行可有较大 stride。
#   qweight/kweight: 连续的 [128]。
#   positions: 连续的 [3,N]，位置值应落在 cache 支持的范围。
#   cache: 连续的 [位置数,128]，每行 cos64 + sin64。
# 输入应位于兼容的同一 CUDA 设备；原代码没有完整的参数检查。
def fused_qk_norm_mrope(q, k, qweight, kweight, positions, cache, eps=1e-6):
    # empty 只分配空间，不初始化数值；kernel 会写完约定范围内的输出。
    # 保持 q/k 各自的形状、设备与 dtype，输出为分别分配的连续张量。
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty(k.shape, device=k.device, dtype=k.dtype)

    # grid=(N,NQ+NK)。例如 N=4、NQ=32、NK=4 时是 (4,36)，共 144 个 program。
    # 这些是逻辑任务，不代表 GPU 一定同时执行全部任务。
    _qk_norm_mrope[(q.shape[0], q.shape[1] // 128 + k.shape[1] // 128)](
        q, k, qweight, kweight, positions, cache, oq, ok,
        q.shape[0],         # N
        q.shape[1] // 128,  # NQ
        k.shape[1] // 128,  # NK
        q.stride(0),       # QS，单位为元素
        k.stride(0),       # KS，单位为元素
        eps,               # EPS，保持与原模型 Norm 一致
        num_warps=4,       # 每个 program 的 warp 配置，不是 program 总数
        enable_fp_fusion=False, # 限制浮点乘加融合；不等于关闭这里的算子融合
    )
    return oq, ok
