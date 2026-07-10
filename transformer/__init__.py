"""
transformer/__init__.py — 包初始化与公共 API 导出

本文件是整个 Transformer 纯 NumPy 实现的入口点。它从所有子模块中导入核心类
和函数，并将它们统一暴露给外部调用者。这样使用者只需 `from transformer import ...`
即可访问全部组件，而无需知道内部的模块结构。

导出的组件涵盖了 Transformer 架构的所有核心构建块：
- 词嵌入（Embedding）与位置编码（PositionalEncoding）
- 多头注意力机制（MultiHeadAttention）与点积注意力（scaled_dot_product_attention）
- 前馈神经网络（FeedForward）
- 层归一化（LayerNorm）与残差连接（ResidualConnection）
- 编码器（Encoder/EncoderLayer）与解码器（Decoder/DecoderLayer）
- 完整的 Transformer Seq2Seq 模型
- 工具函数集合（utils）
"""

# 导入嵌入模块：词向量表示与正弦位置编码
from transformer.embedding import Embedding, PositionalEncoding

# 导入注意力模块：单注意力头与多头注意力
from transformer.attention import MultiHeadAttention, scaled_dot_product_attention

# 导入前馈网络模块：逐位置两层线性变换 + ReLU
from transformer.feed_forward import FeedForward

# 导入归一化模块：LayerNorm 与 Pre-LN 残差连接
from transformer.normalization import LayerNorm, ResidualConnection

# 导入编码器模块：单层编码器与完整编码器堆栈
from transformer.encoder import Encoder, EncoderLayer

# 导入解码器模块：单层解码器与完整解码器堆栈
from transformer.decoder import Decoder, DecoderLayer

# 导入完整 Transformer 模型
from transformer.model import Transformer

# 导入工具模块（mask 生成函数）
from transformer import utils

# __all__ 定义了 `from transformer import *` 时会导出哪些名称
__all__ = [
    'Embedding',            # 词嵌入层 (vocab_size → d_model)
    'PositionalEncoding',   # 正弦位置编码（无参数）
    'MultiHeadAttention',   # 多头注意力机制
    'scaled_dot_product_attention',  # 原始缩放点积注意力函数
    'FeedForward',          # 逐位置前馈网络
    'LayerNorm',            # 层归一化
    'ResidualConnection',   # Pre-LN 残差连接
    'Encoder',              # 完整编码器
    'EncoderLayer',         # 单个编码器层
    'Decoder',              # 完整解码器
    'DecoderLayer',         # 单个解码器层
    'Transformer',          # 完整 Seq2Seq Transformer
    'utils',                # 工具函数模块
]
