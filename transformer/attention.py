"""
transformer/attention.py — 注意力机制（Attention Mechanism）

注意力机制是 Transformer 的核心。它让模型在处理序列中的每个位置时，
能够"关注"序列中所有其他位置的信息，并根据相关性分配不同权重。

本模块包含两个核心组件：
  1. scaled_dot_product_attention — 基础的缩放点积注意力函数
  2. MultiHeadAttention — 多头注意力机制（Multi-Head Attention）

== 缩放点积注意力的数学原理 ==

  Attention(Q, K, V) = softmax(Q · K^T / sqrt(d_k)) · V

  - Q (Query):    查询向量，表示"我要找什么"
  - K (Key):      键向量，表示"我有什么"
  - V (Value):    值向量，表示"我的内容是什么"
  - sqrt(d_k):    缩放因子，防止点积结果过大导致 softmax 饱和

  直观理解：Q 和 K 的点积衡量了两个位置之间的"相关性"，值越大说明越
  相关。经过 softmax 后变为概率分布（和为 1），再用这些权重对 V 加权求和，
  得到融合了上下文信息的输出。

== 多头注意力的动机 ==

  单一注意力头只能学习一种关注模式。多头注意力将 Q, K, V 投影到多个
  低维子空间（每个头 d_k = d_model / n_heads），在每个子空间中独立
  计算注意力，最后将结果拼接并再次投影。这样模型可以同时在不同的表示
  子空间上关注不同位置的不同信息（比如一个头关注语法关系，另一个头
  关注语义相关性）。
"""
import numpy as np


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    缩放点积注意力（Scaled Dot-Product Attention）。

    这是整个注意力机制的核心计算。它接收 Query、Key、Value 三个张量，
    计算注意力分数，应用 softmax 归一化，然后对 Value 进行加权求和。

    计算流程：
      1. 计算注意力分数: scores = Q · K^T / sqrt(d_k)
         形状: (batch, n_heads, seq_q, seq_k)
      2. 应用 mask（可选）: 将 mask 为 0 的位置分数设为 -1e9（近似 -∞）
         这样 softmax 后这些位置的权重趋近于 0
      3. Softmax 归一化: weights = softmax(scores, axis=-1)
         每个 query 位置对所有 key 位置的权重之和为 1
      4. 加权求和: output = weights · V
         对每个 query 位置，根据其注意力权重对 Value 加权求和

    数值稳定性：
      softmax 计算前减去最大值 (scores - max)，防止 exp 溢出。
      这在数学上等价于原 softmax（因为 softmax(x-c) = softmax(x)），
      但避免了大数 exp 导致 Inf 的问题。

    Args:
        q:    (batch, n_heads, seq_len_q, d_k) 查询张量
        k:    (batch, n_heads, seq_len_k, d_k) 键张量
        v:    (batch, n_heads, seq_len_k, d_k) 值张量
        mask: 可选，(batch, 1, 1, seq_len_k) 或 (batch, 1, seq_len_q, seq_len_k)
              值为 0 的位置会被屏蔽

    Returns:
        output: (batch, n_heads, seq_len_q, d_k) 注意力加权输出
        weights: (batch, n_heads, seq_len_q, seq_len_k) 注意力权重矩阵
    """
    # 获取 Key 的维度 d_k，用于缩放
    d_k = q.shape[-1]

    # 步骤 1: 计算点积 Q · K^T
    # einsum 表示: 对最后一个维度 d_k 做内积
    # 结果: (batch, n_heads, seq_q, seq_k)
    scores = np.einsum('bhqd,bhkd->bhqk', q, k) / np.sqrt(d_k)

    # 步骤 2: 应用 mask
    # mask == 0 的位置替换为 -1e9，softmax 后权重 ≈ 0
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # 步骤 3: Softmax 归一化（沿 key 序列维度）
    # 减去最大值防止溢出
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 步骤 4: 用注意力权重对 Value 加权求和
    # einsum 表示: weights(b,h,q,k) × v(b,h,k,d) → 对 k 维度求和
    output = np.einsum('bhqk,bhkd->bhqd', weights, v)

    return output, weights


class MultiHeadAttention:
    """
    多头注意力（Multi-Head Attention）实现。

    架构流程：
      1. 将输入分别通过三个线性层投影为 Q, K, V（每个都是 d_model → d_model）
      2. 将 Q, K, V 切分为 n_heads 个头，每个头的维度为 d_k = d_model // n_heads
      3. 在每个头上独立计算 scaled_dot_product_attention
      4. 将所有头的输出拼接（concatenate）回 d_model 维度
      5. 通过一个输出线性层进行最终投影

    参数说明：
      - w_q, b_q: Query 投影层的权重和偏置，形状 (d_model, d_model)
      - w_k, b_k: Key 投影层的权重和偏置
      - w_v, b_v: Value 投影层的权重和偏置
      - w_o, b_o: 输出投影层的权重和偏置
      所有参数均使用 Xavier 均匀分布初始化。

    反向传播：
      从输出层反向经过：输出投影 → reshape → scaled_dot_attention 反向 →
      Q/K/V 线性投影 → 返回给前一层。Q/K/V 的反向梯度分别返回，以便
      在解码器中处理 self-attention（Q=K=V）和 cross-attention（Q≠K≠V）
      两种情况。
    """

    def __init__(self, d_model, n_heads):
        """
        初始化多头注意力。

        Args:
            d_model:  模型维度
            n_heads:  注意力头数（必须整除 d_model）
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # Xavier 均匀分布初始化 Q, K, V, Output 四个线性投影的权重和偏置
        scale_q = np.sqrt(2.0 / (d_model + d_model))
        scale_out = np.sqrt(2.0 / (d_model + d_model))

        self.w_q = np.random.uniform(-scale_q, scale_q, (d_model, d_model))
        self.b_q = np.zeros(d_model)
        self.w_k = np.random.uniform(-scale_q, scale_q, (d_model, d_model))
        self.b_k = np.zeros(d_model)
        self.w_v = np.random.uniform(-scale_q, scale_q, (d_model, d_model))
        self.b_v = np.zeros(d_model)
        self.w_o = np.random.uniform(-scale_out, scale_out, (d_model, d_model))
        self.b_o = np.zeros(d_model)

        # 为所有参数创建零梯度数组
        for name in ['w_q', 'b_q', 'w_k', 'b_k', 'w_v', 'b_v', 'w_o', 'b_o']:
            setattr(self, f'grad_{name}', np.zeros_like(getattr(self, name)))

        # 缓存前向传播的中间结果，供反向传播使用
        self._cache = {}

    def _linear(self, x, w, b):
        """
        线性变换：y = x @ w + b。

        Args:
            x: (batch, seq, d_in) 输入
            w: (d_in, d_out) 权重矩阵
            b: (d_out,) 偏置向量

        Returns:
            (batch, seq, d_out) 变换后的输出
        """
        return x @ w + b

    def forward(self, q_in, k_in, v_in, mask=None):
        """
        多头注意力前向传播。

        详细流程：
          1. 线性投影: Q, K, V 分别经过独立线性层
          2. 多头分割: reshape 为 (batch, seq, n_heads, d_k) 再 transpose
             为 (batch, n_heads, seq, d_k)，使注意力计算可以在 n_heads 维度上并行
          3. 缩放点积注意力: 调用 scaled_dot_product_attention
          4. 合并多头: transpose 回 (batch, seq, n_heads, d_k) 后 reshape
             为 (batch, seq, d_model)
          5. 输出投影: 经过最后一层线性变换

        Args:
            q_in:   (batch, seq_q, d_model) 查询输入
            k_in:   (batch, seq_k, d_model) 键输入
            v_in:   (batch, seq_k, d_model) 值输入
            mask:   可选，注意力掩码

        Returns:
            output: (batch, seq_q, d_model) 多头注意力输出
        """
        batch_size = q_in.shape[0]

        # 步骤 1: 线性投影到 Q, K, V
        # 形状: (batch, seq, d_model)
        q = self._linear(q_in, self.w_q, self.b_q)
        k = self._linear(k_in, self.w_k, self.b_k)
        v = self._linear(v_in, self.w_v, self.b_v)

        # 步骤 2: 重塑为多头格式
        # (batch, seq, d_model) → (batch, seq, n_heads, d_k) → (batch, n_heads, seq, d_k)
        # transpose(0, 2, 1, 3) 将 n_heads 维度移到第 2 维
        q = q.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # 步骤 3: 计算缩放点积注意力
        # attn_out: (batch, n_heads, seq, d_k)
        # attn_w:   (batch, n_heads, seq, seq)
        attn_out, attn_w = scaled_dot_product_attention(q, k, v, mask)

        # 步骤 4: 合并多头输出
        # (batch, n_heads, seq, d_k) → (batch, seq, n_heads, d_k) → (batch, seq, d_model)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        # 步骤 5: 输出投影
        output = self._linear(attn_out, self.w_o, self.b_o)

        # 缓存所有中间变量
        self._cache = {
            'q_in': q_in, 'k_in': k_in, 'v_in': v_in,
            'q': q, 'k': k, 'v': v,
            'attn_out': attn_out, 'attn_w': attn_w,
            'batch_size': batch_size,
        }

        return output

    def backward(self, grad_output):
        """
        多头注意力的反向传播。

        反向顺序与前向相反：
          1. 输出投影的反向 → 得到合并多头后的梯度 d_attn_out
          2. reshape 回多头格式 → (batch, n_heads, seq, d_k)
          3. 缩放点积注意力的反向 → 得到 d_q, d_k, d_v（多头格式）
          4. reshape 回序列格式 → (batch, seq, d_model)
          5. Q/K/V 线性投影的反向 → 累积各参数的梯度
          6. 返回给前一层的梯度 (grad_q, grad_k, grad_v)

        Args:
            grad_output: (batch, seq, d_model) 来自上层的梯度

        Returns:
            grad_q, grad_k, grad_v: 分别回传到 Q/K/V 前层的梯度
        """
        cache = self._cache
        batch_size = cache['batch_size']

        # 步骤 1: 输出投影的反向
        # d_attn_out: (batch, seq, d_model)
        d_attn_out = grad_output @ self.w_o.T
        self.grad_w_o = np.einsum('bij,bik->jk', cache['attn_out'], grad_output)
        self.grad_b_o = grad_output.sum(axis=(0, 1))

        # 步骤 2: reshape 回多头格式
        # (batch, seq, d_model) → (batch, n_heads, seq, d_k)
        d_attn_out_heads = d_attn_out.reshape(
            batch_size, -1, self.n_heads, self.d_k
        ).transpose(0, 2, 1, 3)

        # 步骤 3: 缩放点积注意力的反向
        d_q, d_k, d_v = self._backward_attention(d_attn_out_heads)

        # 步骤 4: reshape 回序列格式
        # (batch, n_heads, seq, d_k) → (batch, seq, d_model)
        d_q = d_q.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        d_k = d_k.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        d_v = d_v.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        # 步骤 5: Q/K/V 线性投影的反向
        q_in = cache['q_in']
        k_in = cache['k_in']
        v_in = cache['v_in']

        self.grad_w_q = np.einsum('bij,bik->jk', q_in, d_q)
        self.grad_b_q = d_q.sum(axis=(0, 1))
        self.grad_w_k = np.einsum('bij,bik->jk', k_in, d_k)
        self.grad_b_k = d_k.sum(axis=(0, 1))
        self.grad_w_v = np.einsum('bij,bik->jk', v_in, d_v)
        self.grad_b_v = d_v.sum(axis=(0, 1))

        # 步骤 6: 返回给前一层的梯度
        grad_q = d_q @ self.w_q.T
        grad_k = d_k @ self.w_k.T
        grad_v = d_v @ self.w_v.T

        return grad_q, grad_k, grad_v

    def _backward_attention(self, d_output):
        """
        缩放点积注意力的反向传播。

        这是整个反向传播中最复杂的部分。根据前向传播的公式：
          output = softmax(Q·K^T/√d_k) · V

        反向推导：
          1. dV = W^T · d_output   （W 是注意力权重）
          2. dW = d_output · V^T   （链式法则）
          3. d_scores = W * (dW - sum(dW * W))  （softmax 的雅可比矩阵）
          4. dQ = d_scores · K / √d_k
          5. dK = d_scores · Q / √d_k

        softmax 反向的特殊性：
          对于 softmax 输出 s = exp(x) / sum(exp(x))，
          ds_i/dx_j = s_i * (δ_ij - s_j)，其中 δ_ij 是 Kronecker delta。
          展开后：dx = s * (dy - sum(dy * s))。

        Args:
            d_output: (batch, n_heads, seq, d_k) 来自上层的梯度

        Returns:
            d_q, d_k, d_v: (batch, n_heads, seq, d_k) Q/K/V 的梯度
        """
        cache = self._cache
        q, k, v = cache['q'], cache['k'], cache['v']
        attn_w = cache['attn_w']
        d_k = q.shape[-1]

        # d_weights: 对注意力权重矩阵的梯度
        # 因为 output = weights @ v，所以 d_weights = d_output @ v^T
        # (batch, n_heads, seq_q, seq_k)
        d_weights = np.einsum('bhqd,bhkd->bhqk', d_output, v)

        # d_v: 对 Value 的梯度
        # 因为 output = weights @ v，所以 d_v = weights^T @ d_output
        d_v = np.einsum('bhqk,bhqd->bhkd', attn_w, d_output)

        # d_scores: 对注意力分数（softmax 前）的梯度
        # 使用 softmax 的导数公式: dx = x * (dy - sum(dy * x))
        # d_weights_diag = sum(d_weights * weights, axis=-1) 是每个 query 位置的加权和
        d_weights_diag = np.einsum('bhqk,bhqd->bhqd', d_weights, attn_w)
        d_scores = attn_w * (d_weights - d_weights_diag)

        # d_q, d_k: 对 Query 和 Key 的梯度
        # 因为 scores = Q · K^T / √d_k，
        # 所以 dQ = d_scores · K / √d_k, dK = d_scores^T · Q / √d_k
        d_q = np.einsum('bhqk,bhkd->bhqd', d_scores, k) / np.sqrt(d_k)
        d_k = np.einsum('bhqk,bhqd->bhkd', d_scores, q) / np.sqrt(d_k)

        return d_q, d_k, d_v

    def update(self, lr):
        """
        使用 SGD 更新所有参数。

        Args:
            lr: 学习率
        """
        for name in ['w_q', 'b_q', 'w_k', 'b_k', 'w_v', 'b_v', 'w_o', 'b_o']:
            grad = getattr(self, f'grad_{name}')
            param = getattr(self, name)
            param -= lr * grad
