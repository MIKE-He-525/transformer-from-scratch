"""
transformer/embedding.py — 词嵌入（Token Embedding）与位置编码（Positional Encoding）

在 Transformer 中，输入由两个部分组成：
  1. Token Embedding：将离散的 token ID 映射为连续的高维向量，使语义相近
     的 token 在向量空间中也更接近。
  2. Positional Encoding：由于 Transformer 不含递归和卷积，模型本身无法感知
     token 在序列中的位置。位置编码将位置信息注入到嵌入向量中。

本模块实现了两种核心组件：
  - Embedding：可学习的词向量表，与标准的 nn.Embedding 类似但手写反向传播
  - PositionalEncoding：固定的正弦/余弦位置编码（无参数）
"""
import numpy as np


class Embedding:
    """
    词嵌入层（Token Embedding）。

    作用：将每个 token ID（整数索引）映射为一个 d_model 维的实值向量。
    嵌入矩阵的形状为 (vocab_size, d_model)，每一行对应一个词表的向量。

    缩放机制：
      嵌入输出会乘以 sqrt(d_model)，这样做的目的是使嵌入值的量级与位置
      编码相匹配，避免嵌入值过小被位置编码"淹没"。在训练初期，这也让
      embedding 的梯度量级与其他层保持一致。

    反向传播：
      forward 时记录被索引的位置，backward 时使用 np.add.at 将梯度
      累积到对应行的嵌入向量上。由于输入 token ID 是整数，梯度不会回传到输入。
    """

    def __init__(self, vocab_size, d_model):
        """
        初始化嵌入层。

        Args:
            vocab_size: 词表大小（即不同 token 的数量）
            d_model:    嵌入维度，也是整个 Transformer 模型的主维度
        """
        self.vocab_size = vocab_size
        self.d_model = d_model

        # 使用 Xavier/Glorot 均匀分布初始化嵌入矩阵
        # 初始化范围 ~ [-sqrt(2/(V+D)), sqrt(2/(V+D))]，使各层梯度量级均衡
        scale = np.sqrt(2.0 / (vocab_size + d_model))
        self.weight = np.random.uniform(-scale, scale, (vocab_size, d_model))

        # 梯度数组，形状与 weight 相同
        self.grad = np.zeros_like(self.weight)

        # 缓存前向传播的输入索引，用于反向传播
        self._cache = None

    def forward(self, x):
        """
        前向传播：查表 + 缩放。

        流程：
          1. 根据输入 x 中的每个 token ID，从嵌入矩阵中取出对应行
          2. 将结果乘以 sqrt(d_model) 进行缩放

        Args:
            x: (batch, seq_len) 的整数数组，元素为 token 索引

        Returns:
            (batch, seq_len, d_model) 的嵌入向量
        """
        # 保存输入供 backward 使用
        self._cache = x

        # 嵌入查表 + 缩放
        return self.weight[x] * np.sqrt(self.d_model)

    def backward(self, grad_output):
        """
        反向传播：累积梯度到嵌入矩阵。

        流程：
          1. 创建零矩阵 grad_weight
          2. 对于前向传播中每个被访问的 token ID，将对应的上游梯度累加到
             grad_weight 的相应行上（使用 np.add.at 处理重复索引）
          3. 除以 sqrt(d_model) 以抵消 forward 中的缩放
          4. 更新 self.grad

        Args:
            grad_output: (batch, seq_len, d_model) 来自上层的梯度

        Returns:
            None（输入是整数索引，不需要回传梯度）
        """
        x = self._cache

        # 创建零梯度矩阵
        grad_weight = np.zeros_like(self.weight)

        # np.add.at 处理重复索引的梯度累积：
        #   当同一个 token 在序列中出现多次时，其梯度会被累加
        #   grad_output / sqrt(d_model) 是 forward 中乘以 sqrt(d_model) 的链式导数
        np.add.at(grad_weight, x, grad_output / np.sqrt(self.d_model))

        self.grad = grad_weight

        # 输入是整数 token ID，无法对整数求导，故返回 None
        return None

    def update(self, lr):
        """
        更新嵌入矩阵参数（简单 SGD）。

        Args:
            lr: 学习率
        """
        self.weight -= lr * self.grad


class PositionalEncoding:
    """
    正弦位置编码（Sinusoidal Positional Encoding）。

    作用：为序列中的每个位置生成一个固定（不可学习）的位置向量，
    并将其加到 token embedding 上，使模型能感知 token 的相对/绝对位置。

    编码公式：
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))    # 偶数维度用正弦
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))    # 奇数维度用余弦

    设计原理：
      - 不同维度使用不同频率的正弦波，低频编码远距离位置关系，
        高频编码近距离位置关系
      - 对于固定偏移 k，PE(pos+k) 可以表示为 PE(pos) 的线性变换，
        这使模型能够学习到相对位置信息
      - 无需额外训练参数，且可以外推到训练时未见过的序列长度

    本实现不含有任何可学习参数，backward 只是将梯度直接透传。
    """

    def __init__(self, d_model, max_len=5000):
        """
        预计算位置编码矩阵。

        Args:
            d_model: 嵌入维度
            max_len: 支持的最大序列长度，预计算所有位置 0~max_len-1 的编码
        """
        self.d_model = d_model
        self.max_len = max_len

        # 创建位置编码矩阵 (max_len, d_model)，初始化为 0
        pe = np.zeros((max_len, d_model))

        # pos 向量: (max_len, 1)，表示每个位置的索引 [0, 1, 2, ..., max_len-1]
        pos = np.arange(max_len).reshape(-1, 1)

        # 频率维度项: (d_model // 2,)
        # div_term[i] = exp(-i * log(10000) / d_model) = 1 / 10000^(i / d_model)
        # 这对应论文公式中的 1 / 10000^(2i / d_model)，其中 i 遍历 0, 2, 4, ...
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        # 偶数索引维度 (0, 2, 4, ...) 使用 sin(pos * div_term)
        pe[:, 0::2] = np.sin(pos * div_term)

        # 奇数索引维度 (1, 3, 5, ...) 使用 cos(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)

        # 保存预计算的位置编码表
        self.pe = pe  # (max_len, d_model)

        # 缓存前向传播的输入，用于反向传播
        self._x = None

    def forward(self, x):
        """
        将位置编码加到输入嵌入上。

        流程：
          1. 根据序列的实际长度 seq_len，从预计算的 pe 中截取前 seq_len 行
          2. 增加 batch 维度，变为 (1, seq_len, d_model)
          3. 与输入 x (batch, seq_len, d_model) 逐元素相加

        Args:
            x: (batch, seq_len, d_model) 嵌入后的 token 向量

        Returns:
            (batch, seq_len, d_model) 加入了位置信息的向量
        """
        self._x = x
        seq_len = x.shape[1]

        # 将 (max_len, d_model) 的位置编码切片为 (seq_len, d_model)
        # 增加 batch 维度 → (1, seq_len, d_model)，通过广播加到每个 batch
        return x + self.pe[np.newaxis, :seq_len, :]

    def backward(self, grad_output):
        """
        反向传播：位置编码无参数，直接透传梯度。

        因为加法操作的导数为 1，且位置编码没有可学习参数，
        所以梯度原封不动地回传。

        Args:
            grad_output: (batch, seq_len, d_model) 来自上层的梯度

        Returns:
            (batch, seq_len, d_model) 透传的梯度
        """
        return grad_output.copy()
