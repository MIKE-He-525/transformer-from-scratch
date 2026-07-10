"""
transformer/utils.py — Mask 工具函数

在 Transformer 中，mask 用于控制注意力机制的可见范围。本模块提供以下三种 mask：

1. Padding Mask（填充掩码）
   - 目的：忽略序列中填充（padding）位置的 token，防止模型关注无效字符。
   - 场景：源序列（encoder）和交叉注意力（decoder cross-attention）中。

2. Causal Mask（因果掩码 / 下三角掩码）
   - 目的：确保解码器在预测第 t 个位置时，只能看到前 t-1 个位置，不能"偷看"未来。
   - 场景：解码器自注意力（decoder self-attention）中。

3. Combined Mask（组合掩码）
   - 目的：将因果掩码与填充掩码相乘，同时防止关注未来和无效位置。
   - 场景：解码器自注意力的完整 mask。

所有 mask 函数的返回值统一为 4D 张量 (batch, 1, seq_a, seq_b)，以便在
scaled_dot_product_attention 中进行广播计算。
"""
import numpy as np


def create_padding_mask(seq, pad_idx=0):
    """
    创建 Padding Mask（填充掩码）。

    在批量训练中，不同序列长度不同，较短的序列需要用特殊 token（通常是 0）
    填充到相同长度。此函数生成一个二进制 mask，标记哪些位置是有效 token，
    哪些是填充。

    工作原理：
      1. 比较序列与 pad_idx，生成布尔数组（True=有效，False=填充）
      2. 转换为 float32（1.0=有效，0.0=填充）
      3. 在轴 1 和 2 上增加维度，以便后续与注意力分数 (batch, heads, seq, seq) 广播

    Args:
        seq:    (batch, seq_len) 的整数张量，表示 token 索引序列
        pad_idx: 用于表示填充的 token 索引，默认为 0

    Returns:
        mask: (batch, 1, 1, seq_len) 的浮点张量
              有效位置值为 1.0，填充位置值为 0.0
              在注意力计算中，0.0 位置对应的分数会被替换为 -1e9（近似负无穷）
    """
    # (batch, seq_len) → 每个位置是否为非填充 token
    mask = (seq != pad_idx).astype(np.float32)
    # 扩展为 (batch, 1, 1, seq_len)，便于与 (batch, n_heads, seq_q, seq_k) 广播
    return mask[:, np.newaxis, np.newaxis, :]


def create_causal_mask(seq_len):
    """
    创建 Causal Mask（因果掩码 / 下三角掩码）。

    因果掩码是一个下三角全 1 矩阵，用于解码器的自注意力层。它的核心作用是
    "屏蔽未来信息"：对于位置 i，只有 j <= i 的位置 j 才是可见的（值为 1），
    j > i 的位置是不可见的（值为 0）。

    举例说明 (seq_len=4)：
        位置0 只能看到 位置0    → [1, 0, 0, 0]
        位置1 可以看到 位置0,1  → [1, 1, 0, 0]
        位置2 可以看到 位置0,1,2 → [1, 1, 1, 0]
        位置3 可以看到 位置0,1,2,3 → [1, 1, 1, 1]

    训练时，解码器的整个目标序列是已知的（teacher forcing），如果没有因果
    掩码，位置 i 可以直接"看到"位置 i+1 的 token，这就相当于作弊。

    Args:
        seq_len: 目标序列的长度

    Returns:
        mask: (1, 1, seq_len, seq_len) 的下三角矩阵
              下三角（含对角线）为 1.0，上三角为 0.0
    """
    # np.tril 提取下三角部分，ones 创建全 1 矩阵
    return np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.float32))


def create_src_mask(src, pad_idx=0):
    """
    创建源序列的 Padding Mask。

    这是 create_padding_mask 的别名函数，语义上明确表示用于编码器输出
    对应的源序列。在编码器的自注意力和解码器的交叉注意力中，都使用此 mask
    来屏蔽源序列中的填充位置。

    Args:
        src:    (batch, src_len) 的源序列 token 索引
        pad_idx: 填充 token 的索引

    Returns:
        mask: (batch, 1, 1, src_len) 的 padding mask
    """
    return create_padding_mask(src, pad_idx)


def create_tgt_mask(tgt, pad_idx=0):
    """
    创建目标序列的组合 Mask（因果掩码 × 填充掩码）。

    解码器的自注意力需要同时满足两个约束：
      1. 因果约束：不能看到未来的 token（来自 causal mask）
      2. 填充约束：不能关注填充位置（来自 padding mask）

    此函数将两种 mask 逐元素相乘，得到最终的组合 mask。相乘的语义是：
    只有当两个 mask 都为 1 时，该位置才可见。

    举例说明：
      假设 tgt = [[1, 2, 0]]（位置 2 是填充）
      causal mask:
          [[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]
      padding mask (broadcast 到 3×3):
          [[1, 1, 0],
           [1, 1, 0],
           [1, 1, 0]]
      组合 mask (逐元素相乘):
          [[1, 0, 0],
           [1, 1, 0],
           [1, 1, 0]]   ← 位置2 被填充掩码屏蔽，即使因果 mask 允许它看到前面

    Args:
        tgt:    (batch, tgt_len) 的目标序列 token 索引
        pad_idx: 填充 token 的索引

    Returns:
        mask: (batch, 1, tgt_len, tgt_len) 的组合 mask
              同时满足因果性和非填充约束
    """
    batch_size, tgt_len = tgt.shape

    # 因果 mask: (1, 1, tgt_len, tgt_len) — 控制"能否看到前面"
    causal = create_causal_mask(tgt_len)

    # 填充 mask: (batch, tgt_len) → (batch, 1, 1, tgt_len) — 控制"哪些位置有效"
    padding = (tgt != pad_idx).astype(np.float32)
    padding = padding[:, np.newaxis, np.newaxis, :]

    # 逐元素相乘 → (batch, 1, tgt_len, tgt_len)
    # causal 会被广播到 (batch, 1, tgt_len, tgt_len)，与 padding 相乘
    mask = causal * padding

    return mask


def create_cross_mask(src, pad_idx=0):
    """
    创建交叉注意力（Cross-Attention）的 Mask。

    在解码器中，交叉注意力层让解码器的每个位置都能"关注"编码器的所有位置。
    但编码器输出中可能包含源序列的填充位置，这些位置不应被关注。
    因此交叉注意力也需要一个 padding mask，其形式与 create_src_mask 相同。

    在 scaled_dot_product_attention(q, k, v, mask) 中：
      - q 来自解码器（query）
      - k, v 来自编码器输出（key, value）
      - mask 就是此函数生成的 padding mask，用于屏蔽编码器中的无效位置

    Args:
        src:    (batch, src_len) 的源序列 token 索引
        pad_idx: 填充 token 的索引

    Returns:
        mask: (batch, 1, 1, src_len) 的 padding mask
    """
    return create_padding_mask(src, pad_idx)
