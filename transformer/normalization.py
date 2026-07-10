"""
transformer/normalization.py — 归一化与残差连接

本模块实现了 Transformer 中的两种关键结构：
  1. LayerNorm: 层归一化，稳定训练过程
  2. ResidualConnection: Pre-LN 残差连接，缓解梯度消失

== 层归一化（Layer Normalization）==

LayerNorm 对每个样本的最后一个维度（特征维度）进行归一化：
  y = gamma * (x - mean) / std + beta

  - mean, std: 在特征维度上计算（不是 batch 维度，这与 BatchNorm 不同）
  - gamma, beta: 可学习的缩放和平移参数
  - eps: 数值稳定性参数，防止除以 0

为什么使用 LayerNorm 而非 BatchNorm：
  - Transformer 处理变长序列，batch 中各序列长度可能不同
  - 归一化应基于单个样本内部的统计量，而非整个 batch
  - LayerNorm 在 NLP 任务中被证明比 BatchNorm 更有效

== Pre-LN 残差连接 ==

Transformer 中有两种残差连接顺序：
  - Post-LN:  x + LayerNorm(sublayer(x))     （原始论文的顺序）
  - Pre-LN:   LayerNorm(x + sublayer(LayerNorm(x)))  （实际训练中更稳定）

本实现采用 Pre-LN 架构，即先归一化再送入子层（注意力或 FFN），
然后残差连接加回原始输入。Pre-LN 的优势：
  - 梯度流动更直接，训练更稳定
  - 允许更深的网络结构
  - 是目前大多数 Transformer 实现的标准做法

ResidualConnection 的工作：
  forward:  norm(x) → sublayer → dropout → x + result
  backward: 梯度分成两路——一路经过子层反向传播，一路直接从残差连接透传
"""
import numpy as np


class LayerNorm:
    """
    层归一化（Layer Normalization）。

    数学公式：
      y = gamma ⊙ (x - E[x]) / sqrt(Var[x] + eps) + beta

    其中：
      - E[x] 和 Var[x] 在最后一个维度（d_model）上计算
      - gamma 是可学习的缩放参数，初始化为全 1
      - beta 是可学习的平移参数，初始化为全 0
      - eps 是数值稳定性小常数（1e-5）

    参数 gamma 和 beta 的意义：
      归一化后的数据均值为 0、方差为 1，这虽然有助于训练稳定，但
      可能表达能力受限。gamma 和 beta 让模型有机会将归一化后的数据
      变换到更合适的分布——如果模型发现"不需要归一化"更好，它可以
      通过参数学习恢复到原始分布。

    反向传播推导：
      对 LayerNorm 的反向传播需要应用链式法则，依次经过：
        1. gamma/beta 参数的梯度
        2. 标准化操作 (x - mean) / std 的梯度
        3. 均值和方差的梯度（它们本身也依赖于输入 x）

      具体公式（设 m = d_model）：
        d_x_norm = grad_output ⊙ gamma
        d_var    = Σ d_x_norm ⊙ (x - mean) ⊙ (-1/2) ⊙ std^(-3)
        d_mean   = Σ d_x_norm ⊙ (-1/std) + d_var ⊙ mean(-2*(x-mean))
        d_x      = d_x_norm / std + d_var ⊙ 2*(x-mean) / m + d_mean / m
    """

    def __init__(self, d_model, eps=1e-5):
        """
        初始化 LayerNorm。

        Args:
            d_model: 特征维度
            eps:     数值稳定性参数，加在方差上防止开根号为 0
        """
        self.d_model = d_model
        self.eps = eps

        # 可学习的缩放参数（初始化为 1，即初始时不做缩放）
        self.gamma = np.ones(d_model)

        # 可学习的平移参数（初始化为 0，即初始时不做平移）
        self.beta = np.zeros(d_model)

        # 梯度数组
        self.grad_gamma = np.zeros(d_model)
        self.grad_beta = np.zeros(d_model)

        # 缓存前向传播的中间结果
        self._cache = {}

    def forward(self, x):
        """
        前向传播：计算均值、方差，进行归一化。

        流程：
          1. 在最后一个维度上计算均值 mean: (batch, seq, 1)
          2. 在最后一个维度上计算方差 var:  (batch, seq, 1)
          3. 标准化: x_norm = (x - mean) / sqrt(var + eps)
          4. 缩放平移: out = gamma * x_norm + beta

        Args:
            x: (batch, seq, d_model) 输入

        Returns:
            (batch, seq, d_model) 归一化后的输出
        """
        # 在最后一个维度（特征维度）上计算统计量
        # keepdims=True 保持维度为 (batch, seq, 1)，方便后续广播
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # 标准化
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # 缩放 + 平移
        out = self.gamma * x_norm + self.beta

        # 缓存所有中间变量
        self._cache = {'x': x, 'mean': mean, 'var': var, 'x_norm': x_norm}

        return out

    def backward(self, grad_output):
        """
        反向传播：计算参数梯度和输入梯度。

        推导过程：
          令 m = d_model（特征维度大小）
          std = sqrt(var + eps)
          std_inv = 1 / std

          (1) gamma 的梯度:
              d_gamma = Σ(grad_output ⊙ x_norm)  对 batch, seq 维度求和

          (2) beta 的梯度:
              d_beta = Σ(grad_output)  对 batch, seq 维度求和

          (3) x_norm 的梯度:
              d_x_norm = grad_output ⊙ gamma

          (4) 方差的梯度（链式法则，var → sqrt(var+eps) → x_norm）:
              d_var = Σ d_x_norm ⊙ (x - mean) ⊙ (-0.5) ⊙ std^(-3)

          (5) 均值的梯度（来自两条路径：直接路径 + 通过方差的间接路径）:
              直接路径: d_x_norm ⊙ (-std_inv)
              间接路径: d_var ⊙ mean(-2*(x - mean))
              d_mean = Σ(直接路径) + 间接路径

          (6) 输入梯度（来自三条路径：通过 x_norm、通过 var、通过 mean）:
              d_x = d_x_norm ⊙ std_inv + d_var ⊙ 2*(x-mean)/m + d_mean/m

        Args:
            grad_output: (batch, seq, d_model) 来自上层的梯度

        Returns:
            (batch, seq, d_model) 回传到输入的梯度
        """
        cache = self._cache
        x = cache['x']
        mean = cache['mean']
        var = cache['var']
        x_norm = cache['x_norm']
        m = self.d_model

        # (1) gamma 的梯度
        self.grad_gamma = np.sum(grad_output * x_norm, axis=(0, 1))

        # (2) beta 的梯度
        self.grad_beta = np.sum(grad_output, axis=(0, 1))

        # 预计算 std 的逆
        std_inv = 1.0 / np.sqrt(var + self.eps)

        # (3) x_norm 的梯度
        d_x_norm = grad_output * self.gamma

        # (4) 方差的梯度
        # std = sqrt(var + eps), 所以 d_std/d_var = 1/(2*std)
        # x_norm = (x - mean) / std, 所以 dx_norm/d_var = (x-mean) * (-1/std^2) * d_std/d_var
        #                                    = (x-mean) * (-0.5) * std^(-3)
        d_var = np.sum(
            d_x_norm * (x - mean) * (-0.5) * std_inv**3,
            axis=-1, keepdims=True
        )

        # (5) 均值的梯度
        # 两条路径：
        #   路径A: x_norm 直接依赖于 mean → d_x_norm * (-std_inv)
        #   路径B: var 依赖于 mean → d_var * mean(-2*(x-mean))
        #     var = mean((x-mean)^2), d_var/d_mean = mean(-2*(x-mean))
        d_mean = (
            np.sum(d_x_norm * (-std_inv), axis=-1, keepdims=True)
            + d_var * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True)
        )

        # (6) 输入梯度
        # 三条路径：
        #   路径A: 通过 x_norm → d_x_norm * std_inv
        #   路径B: 通过 var  → d_var * 2*(x-mean) / m
        #   路径C: 通过 mean → d_mean / m
        grad_input = (
            d_x_norm * std_inv
            + d_var * 2.0 * (x - mean) / m
            + d_mean / m
        )

        return grad_input

    def update(self, lr):
        """
        使用 SGD 更新 gamma 和 beta 参数。

        Args:
            lr: 学习率
        """
        self.gamma -= lr * self.grad_gamma
        self.beta -= lr * self.grad_beta


class ResidualConnection:
    """
    Pre-LayerNorm 残差连接（Pre-LN Residual Connection）。

    这是 Transformer 中每个子层（自注意力 / FFN / 交叉注意力）与残差连接
    的组合封装。Pre-LN 的顺序为：

        x → LayerNorm → Sublayer → Dropout → + → output
                                    ↑         |
                                    └─────────┘ (残差连接)

    即：output = x + Dropout(Sublayer(LayerNorm(x)))

    与原始论文 Post-LN（先残差后归一化）相比，Pre-LN 的优点：
      - 梯度可以直接通过残差连接流动，不经过归一化
      - 训练初期更稳定，不容易发散
      - 支持训练更深的网络

    本类支持两种子层调用方式：
      1. 注意力层: forward(x, sublayer, q, k, v, mask)  — 4+ 参数
      2. FFN:      forward(x, sublayer)                  — 2 参数
    通过参数数量自动判断调用方式。
    """

    def __init__(self, d_model, dropout=0.0):
        """
        初始化残差连接组件。

        Args:
            d_model:    特征维度
            dropout:    Dropout 比率（应用于子层输出）
        """
        # Pre-LN 中的层归一化
        self.norm = LayerNorm(d_model, eps=1e-5)

        # Dropout 比率
        self.dropout = dropout

        # 缓存子层引用，用于反向传播
        self._sublayer = None

        # 标记子层是否为 QKV 风格（注意力层）
        self._is_qkv = False

    def forward(self, x, sublayer, *args):
        """
        Pre-LN 残差连接前向传播。

        调用方式：
          注意力: rc.forward(x, attention_layer, q, k, v, mask)
          FFN:    rc.forward(x, ffn_layer)

        流程：
          1. 归一化: x_normed = LayerNorm(x)
          2. 子层计算: sub_out = Sublayer(x_normed, ...)
          3. Dropout: 对 sub_out 应用 Inverted Dropout
          4. 残差: output = x + sub_out

        Args:
            x:        (batch, seq, d_model) 输入
            sublayer: 子层模块（有 forward 方法的对象）
            *args:    子层 forward 的额外参数

        Returns:
            (batch, seq, d_model) 残差连接输出
        """
        self._sublayer = sublayer

        # 根据参数数量判断子层类型
        if len(args) >= 4:
            # 注意力层: args = (q, k, v, mask)
            self._is_qkv = True
            q, k, v, mask = args[0], args[1], args[2], args[3]

            # Pre-LN: 先归一化
            x_normed = self.norm.forward(x)

            # 子层计算
            sub_out = sublayer.forward(x_normed, k, v, mask)
        else:
            # FFN: args = ()
            self._is_qkv = False
            x_normed = self.norm.forward(x)
            sub_out = sublayer.forward(x_normed)

        # 应用 Inverted Dropout
        if self.dropout > 0:
            # 生成随机 mask
            self._mask = (np.random.rand(*sub_out.shape) > self.dropout).astype(np.float32)
            # 缩放: 保留值放大 1/(1-dropout) 倍
            sub_out = sub_out * self._mask / (1.0 - self.dropout)

        # 缓存原始输入用于反向
        self._cache_x = x

        # 残差连接: 原始输入 + 子层输出
        return x + sub_out

    def backward(self, grad_output):
        """
        Pre-LN 残差连接反向传播。

        梯度分成两路：
          路径 1: grad_output → Dropout 反向 → 子层反向 → norm 反向 → 残差合并
          路径 2: grad_output 直接（通过残差连接）→ 与路径 1 相加

        对于注意力子层，backward 返回 (grad_q, grad_k, grad_v) 三个梯度，
        我们只取 grad_q（即经过 norm 后的子层输入的梯度），然后通过 norm
        反向传播到 x。

        Args:
            grad_output: (batch, seq, d_model) 来自上层的梯度

        Returns:
            (batch, seq, d_model) 回传到输入的梯度
        """
        # Dropout 反向（与 forward 相同的 mask 和缩放）
        if self.dropout > 0:
            grad_sub = grad_output * self._mask / (1.0 - self.dropout)
        else:
            grad_sub = grad_output.copy()

        # 梯度通过子层反向传播
        if self._is_qkv:
            # 注意力层返回三个梯度，取 grad_q
            q_grad, k_grad, v_grad = self._sublayer.backward(grad_sub)
        else:
            q_grad = self._sublayer.backward(grad_sub)

        # 梯度通过 LayerNorm 反向传播
        # q_grad 就是经过子层后对 x_normed 的梯度
        grad_normed = q_grad

        # LayerNorm 反向 + 残差连接的直接透传
        # self.norm.backward 计算 norm 的梯度
        # grad_output 是残差连接中直接透传的梯度（因为 output = x + sub_out）
        grad_x = self.norm.backward(grad_normed) + grad_output

        return grad_x

    def update(self, lr):
        """
        更新 LayerNorm 的参数。

        Args:
            lr: 学习率
        """
        self.norm.update(lr)
