"""
transformer/encoder.py — 编码器（Encoder）

编码器是 Transformer 处理输入序列的组件。它将源序列（如一句话的 token 序列）
转换为一组连续的隐状态表示，供解码器使用。

== 编码器的架构 ==

完整的编码器由以下组件堆叠而成：

    输入 token IDs
         │
         ▼
    ┌─────────────────┐
    │   Token         │   词嵌入层：ID → d_model 维向量
    │   Embedding     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Positional     │   正弦位置编码：注入位置信息
    │  Encoding       │
    └────────┬────────┘
             │
             ▼
    ╔══════════════════════════════════╗
    ║  ┌───────────────────────────┐  ║
    ║  │  Encoder Layer 1          │  ║
    ║  │  ┌─────────────────────┐  │  ║
    ║  │  │ Multi-Head Attention│  │  ║  自注意力：序列内 token 交互
    ║  │  └─────────┬──────────┘  │  ║
    ║  │  ┌─────────┴──────────┐  │  ║
    ║  │  │  Residual + Norm  │  │  ║  Pre-LN 残差连接
    ║  │  └─────────┬──────────┘  │  ║
    ║  │  ┌─────────┴──────────┐  │  ║
    ║  │  │  Feed-Forward Net │  │  ║  逐位置非线性变换
    ║  │  └─────────┬──────────┘  │  ║
    ║  │  ┌─────────┴──────────┐  │  ║
    ║  │  │  Residual + Norm  │  │  ║  Pre-LN 残差连接
    ║  │  └───────────────────┘  │  ║
    ║  ╚══════════════════════════╝  ║
    ║  ...（可堆叠 N 层）               ║
    ║  ┌───────────────────────────┐  ║
    ║  │  Encoder Layer N          │  ║
    ║  └───────────────────────────┘  ║
    ╚══════════════════════════════════╝
             │
             ▼
    编码器输出 (batch, src_len, d_model)

== 自注意力（Self-Attention）的含义 ==

"自"表示 Q、K、V 都来自同一个输入序列。每个位置的 token 都能"关注"
序列中所有其他位置的 token，学习它们之间的依赖关系。这与 RNN 的顺序处理
不同，注意力是并行计算所有位置之间的关系。

== Mask 在编码器中的作用 ==

编码器使用 padding mask 来屏蔽源序列中的填充位置。
在 self-attention 中，mask 为 0 的位置会被赋予接近 0 的注意力权重，
确保填充 token 不会影响有效 token 的表示。
"""
import numpy as np

# 导入多头注意力模块
from transformer.attention import MultiHeadAttention

# 导入前馈网络模块
from transformer.feed_forward import FeedForward

# 导入归一化与残差连接模块
from transformer.normalization import ResidualConnection


class EncoderLayer:
    """
    单个编码器层（Encoder Layer）。

    每个编码器层由两个子层组成：
      子层 1: 多头自注意力（Multi-Head Self-Attention）+ Pre-LN 残差连接
      子层 2: 逐位置前馈网络（Position-wise Feed-Forward）+ Pre-LN 残差连接

    数据流：
      x → LayerNorm → SelfAttention → Dropout → +x → x'
      x' → LayerNorm → FeedForward → Dropout → +x' → output

    在自注意力子层中，Q = K = V = x（都来自同一个输入）。
    """

    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.0):
        """
        初始化编码器层。

        Args:
            d_model:    模型维度
            n_heads:    注意力头数
            hidden_dim: FFN 的中间层维度
            dropout:    Dropout 比率
        """
        # 自注意力层：Q, K, V 都来自同一个输入
        self.self_attn = MultiHeadAttention(d_model, n_heads)

        # 前馈网络层
        self.ffn = FeedForward(d_model, hidden_dim, dropout)

        # 两个残差连接（每个包含 Pre-LN 和 Dropout）
        self.res1 = ResidualConnection(d_model, dropout)
        self.res2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask):
        """
        编码器层的前向传播。

        Args:
            x:     (batch, src_len, d_model) 输入表示
            mask:  (batch, 1, 1, src_len) padding mask，屏蔽填充位置

        Returns:
            (batch, src_len, d_model) 编码后的表示
        """
        # 子层 1: 自注意力 + 残差连接
        # x 同时作为 Q, K, V 传入
        x = self.res1.forward(x, self.self_attn, x, x, x, mask)

        # 子层 2: 前馈网络 + 残差连接
        x = self.res2.forward(x, self.ffn)

        return x

    def backward(self, grad_output):
        """
        编码器层的反向传播。

        按照前向传播的逆序，依次反向经过：
          FFN 残差连接 → 自注意力残差连接

        Args:
            grad_output: (batch, src_len, d_model) 来自上层（或下一个编码器层）的梯度

        Returns:
            (batch, src_len, d_model) 回传到输入的梯度
        """
        # 逆序 1: FFN 残差连接反向
        grad = self.res2.backward(grad_output)

        # 逆序 2: 自注意力残差连接反向
        grad = self.res1.backward(grad)

        return grad

    def update(self, lr):
        """
        更新编码器层的所有参数。

        Args:
            lr: 学习率
        """
        self.self_attn.update(lr)
        self.ffn.update(lr)
        self.res1.update(lr)
        self.res2.update(lr)


class Encoder:
    """
    完整编码器（Full Encoder）。

    完整编码器由以下组件组成：
      1. Token Embedding: 将输入 token ID 序列映射为嵌入向量
      2. Positional Encoding: 添加正弦位置编码
      3. N 个编码器层: 逐层堆叠，每层进行自注意力和 FFN 变换

    工作流程：
      token IDs → Embedding → Positional Encoding → Layer 1 → ... → Layer N → output

    随着层数加深，表示逐步从词法信息（单个 token 的含义）抽象为语义信息
    （token 之间的关系和上下文含义）。
    """

    def __init__(self, vocab_size, d_model, n_layers, n_heads, hidden_dim, max_len=5000, dropout=0.0):
        """
        初始化完整编码器。

        Args:
            vocab_size: 源语言词表大小
            d_model:    模型维度
            n_layers:   编码器层数
            n_heads:    注意力头数
            hidden_dim: FFN 中间层维度
            max_len:    最大序列长度（用于位置编码）
            dropout:    Dropout 比率
        """
        # 延迟导入，避免循环依赖
        from transformer.embedding import Embedding, PositionalEncoding

        # Token 嵌入层
        self.emb = Embedding(vocab_size, d_model)

        # 位置编码层
        self.pos_enc = PositionalEncoding(d_model, max_len)

        # 堆叠 N 个编码器层
        self.layers = [
            EncoderLayer(d_model, n_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ]

    def forward(self, x, mask):
        """
        编码器前向传播。

        流程：
          1. 词嵌入: token IDs → 嵌入向量
          2. 位置编码: 嵌入向量 + 位置信息
          3. 逐层通过 N 个编码器层

        Args:
            x:     (batch, src_len) 源序列的 token ID
            mask:  (batch, 1, 1, src_len) padding mask

        Returns:
            (batch, src_len, d_model) 编码后的序列表示
        """
        # 步骤 1: 词嵌入
        x = self.emb.forward(x)

        # 步骤 2: 位置编码
        x = self.pos_enc.forward(x)

        # 步骤 3: 逐层编码器
        for layer in self.layers:
            x = layer.forward(x, mask)

        return x

    def backward(self, grad_output):
        """
        编码器反向传播。

        按照前向传播的逆序：
          Layer N → ... → Layer 1 → Positional Encoding → Embedding

        Args:
            grad_output: (batch, src_len, d_model) 来自解码器或损失函数的梯度
        """
        # 逆序通过所有编码器层
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

        # 位置编码反向（无参数，直接透传）
        grad_output = self.pos_enc.backward(grad_output)

        # 嵌入层反向（累积嵌入矩阵梯度）
        self.emb.backward(grad_output)

    def update(self, lr):
        """
        更新编码器的所有参数。

        Args:
            lr: 学习率
        """
        # 更新嵌入层
        self.emb.update(lr)

        # 更新所有编码器层
        for layer in self.layers:
            layer.update(lr)
