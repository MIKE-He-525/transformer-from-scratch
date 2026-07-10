"""
transformer/feed_forward.py — 逐位置前馈神经网络（Position-wise Feed-Forward Network）

前馈网络（FFN）是 Transformer 中每个编码器/解码器层内的另一个核心组件。
它紧跟在多头注意力层之后，对序列中每个位置的向量独立地进行非线性变换。

架构：
  FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂

  - W₁: (d_model, hidden_dim)  第一层线性变换，升维到 hidden_dim
  - b₁: (hidden_dim,)          第一层偏置
  - ReLU: 激活函数，引入非线性
  - Dropout: 随机丢弃部分激活值，防止过拟合
  - W₂: (hidden_dim, d_model)  第二层线性变换，降维回 d_model
  - b₂: (d_model,)             第二层偏置

设计原理：
  - 两层线性变换之间夹一个 ReLU 激活函数，使模型能学习复杂的非线性映射
  - hidden_dim 通常远大于 d_model（如 2048 vs 512），提供一个"宽"的中间表示空间
  - 每个位置的向量独立经过相同的 FFN，不与其他位置交互（与注意力不同）
"""
import numpy as np


class FeedForward:
    """
    逐位置前馈神经网络。

    计算公式：FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂

    参数初始化：
      - W₁ 和 W₂ 使用 Xavier/Glorot 均匀分布初始化
      - 偏置初始化为零

    反向传播：
      按照前向顺序的逆序计算梯度：
        1. 第二线性层反向: d_h_relu = grad_output @ W₂^T
        2. Dropout mask 应用（训练时）
        3. ReLU 反向: d_h = d_h_relu ⊙ (h > 0)
        4. 第一线性层反向: 累积 W₁, b₁ 的梯度
        5. 回传: grad_input = d_h @ W₁^T
    """

    def __init__(self, d_model, hidden_dim, dropout=0.0):
        """
        初始化前馈网络。

        Args:
            d_model:    输入/输出维度
            hidden_dim: 中间层维度（通常 > d_model）
            dropout:    Dropout 比率，0.0 表示不使用 Dropout
        """
        self.dropout = dropout

        # Xavier 均匀分布初始化第一层参数
        # 范围: [-sqrt(2/(d_model+hidden_dim)), sqrt(2/(d_model+hidden_dim))]
        scale1 = np.sqrt(2.0 / (d_model + hidden_dim))
        self.w1 = np.random.uniform(-scale1, scale1, (d_model, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        # Xavier 均匀分布初始化第二层参数
        # 范围: [-sqrt(2/(hidden_dim+d_model)), sqrt(2/(hidden_dim+d_model))]
        scale2 = np.sqrt(2.0 / (hidden_dim + d_model))
        self.w2 = np.random.uniform(-scale2, scale2, (hidden_dim, d_model))
        self.b2 = np.zeros(d_model)

        # 为所有参数创建零梯度数组
        for name in ['w1', 'b1', 'w2', 'b2']:
            setattr(self, f'grad_{name}', np.zeros_like(getattr(self, name)))

        # 缓存前向传播的中间结果
        self._cache = {}

    def forward(self, x):
        """
        前向传播。

        流程：
          1. 第一线性层: h = x @ W₁ + b₁    （升维）
          2. ReLU 激活:  h_relu = max(0, h)
          3. Dropout:    随机将部分值置 0，剩余值除以 (1-dropout) 进行缩放
                         这是"Inverted Dropout"，保证训练和推理时期望值一致
          4. 第二线性层: out = h_relu @ W₂ + b₂  （降维）

        Args:
            x: (batch, seq, d_model) 输入张量

        Returns:
            (batch, seq, d_model) 前馈网络输出
        """
        # 第一线性层 + ReLU
        h = x @ self.w1 + self.b1          # (batch, seq, hidden_dim)
        h_relu = np.maximum(0, h)           # ReLU: 负数置零

        # Inverted Dropout（训练时）
        if self.dropout > 0:
            # 生成与 h_relu 同形状的随机掩码，值 > dropout 的保留（1），否则丢弃（0）
            self._mask = (np.random.rand(*h_relu.shape) > self.dropout).astype(np.float32)
            # 将保留的值放大 1/(1-dropout) 倍，保持期望值不变
            h_relu = h_relu * self._mask / (1.0 - self.dropout)
        else:
            # 不使用 dropout 时，mask 全为 1
            self._mask = np.ones_like(h_relu)

        # 第二线性层
        out = h_relu @ self.w2 + self.b2   # (batch, seq, d_model)

        # 缓存所有中间值，用于反向传播
        self._cache = {'x': x, 'h': h, 'h_relu': h_relu, 'mask': self._mask}

        return out

    def backward(self, grad_output):
        """
        反向传播。

        按照前向传播的逆序计算梯度：
          grad_output → 第二线性层反向 → Dropout 反向 → ReLU 反向
          → 第一线性层反向 → 回传给前一层

        Args:
            grad_output: (batch, seq, d_model) 来自上层的梯度

        Returns:
            (batch, seq, d_model) 回传到前一层的梯度
        """
        cache = self._cache
        x = cache['x']
        h_relu = cache['h_relu']
        mask = cache['mask']

        # === 步骤 1: 第二线性层反向 ===
        # 链式法则: d_h_relu = grad_output @ W₂^T
        d_h_relu = grad_output @ self.w2.T  # (batch, seq, hidden_dim)

        # 应用 Dropout mask（与 forward 中相同的缩放）
        if self.dropout > 0:
            d_h_relu = d_h_relu * mask / (1.0 - self.dropout)

        # 累积 W₂, b₂ 的梯度
        # einsum 'bij,bik->jk' 对 batch 和 seq 维度求和: Σ Σ (h_relu_i * grad_j)
        self.grad_w2 = np.einsum('bij,bik->jk', h_relu, grad_output)
        # 偏置梯度: 对 batch 和 seq 维度求和
        self.grad_b2 = grad_output.sum(axis=(0, 1))

        # === 步骤 2: ReLU 反向 ===
        # ReLU 的导数: 输入 > 0 时导数为 1，否则为 0
        # 使用 h（ReLU 前的值）判断，而非 h_relu
        d_h = d_h_relu * (cache['h'] > 0).astype(np.float32)

        # === 步骤 3: 第一线性层反向 ===
        # 累积 W₁, b₁ 的梯度
        self.grad_w1 = np.einsum('bij,bik->jk', x, d_h)
        self.grad_b1 = d_h.sum(axis=(0, 1))

        # === 步骤 4: 回传到前一层 ===
        # grad_input = d_h @ W₁^T
        grad_input = d_h @ self.w1.T
        return grad_input

    def update(self, lr):
        """
        使用 SGD 更新所有参数。

        Args:
            lr: 学习率
        """
        for name in ['w1', 'b1', 'w2', 'b2']:
            param = getattr(self, name)
            grad = getattr(self, f'grad_{name}')
            param -= lr * grad
