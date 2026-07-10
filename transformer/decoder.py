"""
transformer/decoder.py — 解码器（Decoder）

解码器是 Transformer 生成输出序列的组件。它接收编码器的输出表示，
并自回归地（autoregressively）生成目标序列的 token。

== 解码器的架构 ==

每个解码器层包含三个子层：

    解码器输入 token IDs
         │
         ▼
    ┌─────────────────┐
    │   Token         │   词嵌入层
    │   Embedding     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Positional     │   位置编码
    │  Encoding       │
    └────────┬────────┘
             │
             ▼
    ╔══════════════════════════════════╗
    ║  ┌───────────────────────────┐  ║
    ║  │  Decoder Layer            │  ║
    ║  │                           │  ║
    ║  │  ┌─────────────────────┐  │  ║
    ║  │  │ Masked Self-Attention│ │  ║  ← 子层 1：带因果 mask
    ║  │  │  (with causal mask) │  │  ║      防止"偷看"未来 token
    ║  │  └─────────┬──────────┘  │  ║
    ║  │  ┌─────────┴──────────┐  │  ║
    ║  │  │ Residual + Norm   │  │  ║
    ║  │  └─────────┬──────────┘  │  ║
    ║  │           │              │  ║
    ║  │  ┌────────┴──────────┐  │  ║
    ║  │  │ Cross-Attention  │  │  ║  ← 子层 2：交叉注意力
    ║  │  │ (encoder-decoder) │  │  ║      Q 来自解码器，K/V 来自编码器
    ║  │  └─────────┬──────────┘  │  ║
    ║  │  ┌─────────┴──────────┐  │  ║
    ║  │  │ Residual + Norm   │  │  ║
    ║  │  └─────────┬──────────┘  │  ║
    ║  │           │              │  ║
    ║  │  ┌────────┴──────────┐  │  ║
    ║  │  │ Feed-Forward Net │  │  ║  ← 子层 3:FFN
    ║  │  └─────────┬──────────┘  │  ║
    ║  │  ┌─────────┴──────────┐  │  ║
    ║  │  │ Residual + Norm   │  │  ║
    ║  │  └───────────────────┘  │  ║
    ║  ╚══════════════════════════╝  ║
    ╚══════════════════════════════════╝
             │
             ▼
    解码器输出 (batch, tgt_len, d_model)

== 三种注意力的区别 ==

  1. 自注意力（Self-Attention）:
     Q = K = V = 解码器自身的输入
     mask = 因果 mask（只能看到过去和当前位置）

  2. 交叉注意力（Cross-Attention）:
     Q = 解码器输出, K = V = 编码器输出
     mask = 源序列 padding mask（屏蔽源序列中的填充）
     作用：让解码器获取源序列的信息

  3. 前馈网络（Feed-Forward）:
     与编码器相同，逐位置独立非线性变换
"""
import numpy as np

# 导入多头注意力模块
from transformer.attention import MultiHeadAttention

# 导入前馈网络模块
from transformer.feed_forward import FeedForward

# 导入归一化与残差连接模块
from transformer.normalization import ResidualConnection


class DecoderLayer:
    """
    单个解码器层（Decoder Layer）。

    每个解码器层包含三个子层：
      子层 1: 带因果 mask 的 masked 自注意力
      子层 2: 交叉注意力（Q 来自解码器，K/V 来自编码器）
      子层 3: 逐位置前馈网络

    每个子层都有 Pre-LN 残差连接。
    """

    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.0):
        """
        初始化解码器层。

        Args:
            d_model:    模型维度
            n_heads:    注意力头数
            hidden_dim: FFN 中间层维度
            dropout:    Dropout 比率
        """
        # 子层 1: Masked 自注意力（防止"偷看"未来）
        self.self_attn = MultiHeadAttention(d_model, n_heads)

        # 子层 2: 交叉注意力（连接编码器和解码器）
        # 也用 MultiHeadAttention，但调用时 Q 和 K/V 来自不同的输入
        self.cross_attn = MultiHeadAttention(d_model, n_heads)

        # 子层 3: 前馈网络
        self.ffn = FeedForward(d_model, hidden_dim, dropout)

        # 三个残差连接组件（每个包含 Pre-LN + Dropout + 残差加法）
        self.res1 = ResidualConnection(d_model, dropout)
        self.res2 = ResidualConnection(d_model, dropout)
        self.res3 = ResidualConnection(d_model, dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        """
        解码器层的前向传播。

        Args:
            x:        (batch, tgt_len, d_model) 解码器输入
            enc_out:  (batch, src_len, d_model) 编码器输出
            src_mask: (batch, 1, 1, src_len) 源序列 padding mask
            tgt_mask: (batch, 1, tgt_len, tgt_len) 目标序列组合 mask（因果 + padding）

        Returns:
            (batch, tgt_len, d_model) 解码后的表示
        """
        # 子层 1: Masked 自注意力
        # Q = K = V = x，使用 tgt_mask（因果 mask）防止关注未来位置
        x = self.res1.forward(x, self.self_attn, x, x, x, tgt_mask)

        # 子层 2: 交叉注意力
        # Q = x（解码器状态），K = V = enc_out（编码器输出）
        # 使用 src_mask 屏蔽源序列中的填充位置
        x = self.res2.forward(x, self.cross_attn, x, enc_out, enc_out, src_mask)

        # 子层 3: 前馈网络
        x = self.res3.forward(x, self.ffn)

        return x

    def backward(self, grad_output):
        """
        解码器层的反向传播。

        按前向的逆序：
          FFN 残差 → 交叉注意力残差 → Masked 自注意力残差

        Args:
            grad_output: (batch, tgt_len, d_model) 来自上层的梯度

        Returns:
            (batch, tgt_len, d_model) 回传到输入的梯度
        """
        # 逆序 1: FFN 残差连接反向
        grad = self.res3.backward(grad_output)

        # 逆序 2: 交叉注意力残差连接反向
        grad = self.res2.backward(grad)

        # 逆序 3: Masked 自注意力残差连接反向
        grad = self.res1.backward(grad)

        return grad

    def update(self, lr):
        """
        更新解码器层的所有参数。

        Args:
            lr: 学习率
        """
        self.self_attn.update(lr)
        self.cross_attn.update(lr)
        self.ffn.update(lr)
        self.res1.update(lr)
        self.res2.update(lr)
        self.res3.update(lr)


class Decoder:
    """
    完整解码器（Full Decoder）。

    完整解码器由以下组件组成：
      1. Token Embedding: 将目标序列 token ID 映射为嵌入向量
      2. Positional Encoding: 添加正弦位置编码
      3. N 个解码器层: 逐层堆叠，每层包含 masked 自注意力、交叉注意力和 FFN

    工作流程：
      tgt token IDs → Embedding → Positional Encoding → Layer 1 → ... → Layer N → output

    与编码器不同，解码器的每层都需要：
      - 目标序列的 mask（因果 + padding）
      - 编码器的输出（用于交叉注意力）
      - 源序列的 mask（用于交叉注意力的 padding 屏蔽）
    """

    def __init__(self, vocab_size, d_model, n_layers, n_heads, hidden_dim, max_len=5000, dropout=0.0):
        """
        初始化完整解码器。

        Args:
            vocab_size: 目标语言词表大小
            d_model:    模型维度
            n_layers:   解码器层数
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

        # 堆叠 N 个解码器层
        self.layers = [
            DecoderLayer(d_model, n_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ]

    def forward(self, x, enc_out, src_mask, tgt_mask):
        """
        解码器前向传播。

        流程：
          1. 词嵌入: 目标 token IDs → 嵌入向量
          2. 位置编码: 嵌入向量 + 位置信息
          3. 逐层通过 N 个解码器层

        Args:
            x:        (batch, tgt_len) 目标序列的 token ID
            enc_out:  (batch, src_len, d_model) 编码器输出
            src_mask: 源序列 padding mask
            tgt_mask: 目标序列组合 mask（因果 + padding）

        Returns:
            (batch, tgt_len, d_model) 解码后的序列表示
        """
        # 步骤 1: 词嵌入
        x = self.emb.forward(x)

        # 步骤 2: 位置编码
        x = self.pos_enc.forward(x)

        # 步骤 3: 逐层解码器
        for layer in self.layers:
            x = layer.forward(x, enc_out, src_mask, tgt_mask)

        return x

    def backward(self, grad_output):
        """
        解码器反向传播。

        逆序通过所有层，最后经过位置编码和嵌入层。

        Args:
            grad_output: (batch, tgt_len, d_model) 来自上层的梯度
        """
        # 逆序通过所有解码器层
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

        # 位置编码反向
        grad_output = self.pos_enc.backward(grad_output)

        # 嵌入层反向
        self.emb.backward(grad_output)

    def update(self, lr):
        """
        更新解码器的所有参数。

        Args:
            lr: 学习率
        """
        # 更新嵌入层
        self.emb.update(lr)

        # 更新所有解码器层
        for layer in self.layers:
            layer.update(lr)
