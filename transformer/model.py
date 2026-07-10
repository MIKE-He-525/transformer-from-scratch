"""
transformer/model.py — 完整的 Transformer Seq2Seq 模型

本模块将编码器（Encoder）和解码器（Decoder）组合成完整的 Transformer
序列到序列（Seq2Seq）模型。

== 完整的 Transformer 架构 ==

    ┌──────────────────────┐     ┌──────────────────────┐
    │       ENCODER        │     │       DECODER        │
    │                      │     │                      │
    │  Token Embedding     │     │  Token Embedding     │
    │  Positional Encoding │     │  Positional Encoding │
    │                      │     │                      │
    │  ┌────────────────┐  │     │  ┌────────────────┐  │
    │  │ Self-Attention │  │     │  │ Masked SA      │  │
    │  │ FFN            │  │     │  │ Cross-Attention│──┼──→ enc_out
    │  │ ... (× N层)    │  │     │  │ FFN            │  │
    │  └────────────────┘  │     │  └────────────────┘  │
    │                      │     │         │            │
    │                      │     │         ▼            │
    └──────────────────────┘     │  Linear Projection   │
                                 │  (d_model → vocab)   │
                                 │         │            │
                                 │         ▼            │
                                 │  Softmax → 概率分布    │
                                 └──────────────────────┘

== 训练流程（Teacher Forcing）==

  1. 编码: 源序列 → 编码器 → 编码器输出 enc_out
  2. 解码: 目标序列（整体作为输入，带因果 mask）
     → 解码器 → 解码器输出
  3. 投影: 解码器输出 → 线性层 → logits (batch, tgt_len, vocab_size)
  4. 损失: logits 与真实目标计算交叉熵
  5. 反向: 计算所有参数的梯度
  6. 更新: SGD + 梯度裁剪更新所有参数

== 梯度裁剪（Gradient Clipping）==

  在训练深层网络（尤其是自研框架中）时，梯度可能变得非常大，导致参数
  更新过大、损失发散。梯度裁剪限制梯度的全局范数（L2 norm）不超过
  一个阈值，从而稳定训练。

  具体方法：
    total_norm = sqrt(Σ ||g||²)  对所有梯度张量
    if total_norm > max_grad_norm:
        g *= max_grad_norm / total_norm

== 交叉熵损失 ==

  Loss = -1/N × Σ log(p(target_token))

  其中 N = batch_size × tgt_len，p(target_token) 是模型对正确 token
  的预测概率。其梯度有简洁的闭式解：
    grad = probs.copy()
    grad[正确 token 位置] -= 1.0
    grad /= N
"""
import numpy as np

# 导入编码器和解码器
from transformer.encoder import Encoder
from transformer.decoder import Decoder


def _collect_grads(module, grads, seen=None):
    """
    递归收集模块及其子模块中所有 grad_* 开头的梯度数组。

    本函数用于在更新参数前，将整个模型树中所有组件的梯度收集到一个
    扁平列表中，以便统一进行梯度裁剪。

    递归策略：
      - 遍历模块的所有属性
      - 如果属性名以 'grad_' 开头且是 np.ndarray，则加入列表
      - 如果属性是非可调用的对象（非方法、非私有属性），则递归进入
      - 如果属性是 list/tuple，则对每个元素递归

    循环引用保护：
      使用 seen 集合记录已访问的模块 id，防止互相引用的对象导致无限递归。

    Args:
        module:  要收集的模块对象（如 Encoder, Decoder, EncoderLayer 等）
        grads:   已收集的梯度列表（会被就地修改）
        seen:    已访问模块的 id 集合（去重，防止循环引用）
    """
    if seen is None:
        seen = set()

    # 记录当前模块的 id，防止重复访问
    mid = id(module)
    if mid in seen:
        return
    seen.add(mid)

    # 遍历模块的所有属性
    for attr_name in dir(module):
        # 跳过私有属性（以 _ 开头）和可调用的方法
        if attr_name.startswith('grad_'):
            g = getattr(module, attr_name)
            if isinstance(g, np.ndarray):
                grads.append(g)
        elif not attr_name.startswith('_') and not callable(getattr(module, attr_name, None)):
            # 非私有、非可调用 → 可能是子模块，递归收集
            val = getattr(module, attr_name)
            if isinstance(val, (list, tuple)):
                # 处理列表/元组中的子模块
                for item in val:
                    if hasattr(item, '__dict__') and not isinstance(item, type):
                        _collect_grads(item, grads, seen)
            elif hasattr(val, '__dict__') and not isinstance(val, type):
                _collect_grads(val, grads, seen)


class Transformer:
    """
    完整的 Transformer Seq2Seq 模型。

    由编码器、解码器和输出线性投影层组成。支持前向传播、反向传播、
    参数更新、自回归生成等功能。

    标准配置（论文原版）：
      d_model=512, n_layers=6, n_heads=8, hidden_dim=2048, dropout=0.1

    本实现使用纯 NumPy 实现所有组件，不含任何深度学习框架依赖，
    适合学习 Transformer 的内部工作原理。
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        hidden_dim=2048,
        dropout=0.1,
        max_len=5000
    ):
        """
        初始化 Transformer 模型。

        Args:
            src_vocab_size: 源语言词表大小（如英文 30000）
            tgt_vocab_size: 目标语言词表大小
            d_model:        模型维度（嵌入维度）
            n_layers:       编码器和解码器各多少层
            n_heads:        多头注意力的头数
            hidden_dim:     FFN 的中间层维度
            dropout:        Dropout 比率
            max_len:        最大序列长度（位置编码表大小）
        """
        self.d_model = d_model

        # 创建编码器和解码器
        self.encoder = Encoder(
            src_vocab_size, d_model, n_layers, n_heads, hidden_dim, max_len, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_layers, n_heads, hidden_dim, max_len, dropout
        )

        # 输出线性投影层：将解码器的 d_model 维输出映射到 tgt_vocab_size 维
        # 即: 解码器隐状态 → 每个 token 的未归一化 logit
        scale = np.sqrt(2.0 / (d_model + tgt_vocab_size))
        self.fc_w = np.random.uniform(-scale, scale, (d_model, tgt_vocab_size))
        self.fc_b = np.zeros(tgt_vocab_size)

        # 投影层的梯度
        self.grad_fc_w = np.zeros_like(self.fc_w)
        self.grad_fc_b = np.zeros_like(self.fc_b)

        # 缓存前向传播的中间结果
        self._cache = {}

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播：源序列 → 编码器 → 解码器 → logits。

        流程：
          1. 编码器处理源序列 → enc_out (batch, src_len, d_model)
          2. 解码器处理目标序列（以 enc_out 作为上下文）→ dec_out
          3. 线性投影: dec_out @ fc_w + fc_b → logits

        Args:
            src:      (batch, src_len) 源序列 token ID
            tgt:      (batch, tgt_len) 目标序列 token ID
            src_mask: 源序列 padding mask
            tgt_mask: 目标序列组合 mask（因果 + padding）

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size) 未归一化的 logit 分数
        """
        # 步骤 1: 编码源序列
        enc_out = self.encoder.forward(src, src_mask)

        # 步骤 2: 解码目标序列
        # 解码器同时接收编码器的输出和 mask
        dec_out = self.decoder.forward(tgt, enc_out, src_mask, tgt_mask)

        # 步骤 3: 线性投影到词表维度
        logits = dec_out @ self.fc_w + self.fc_b

        # 缓存解码器输出，供反向传播使用
        self._cache = {'dec_out': dec_out}

        return logits

    def backward(self, grad_logits):
        """
        反向传播（仅包含输出投影层和解码器的反向）。

        流程：
          1. 计算输出投影层的梯度 (fc_w, fc_b)
          2. 将梯度传回解码器
          3. 解码器内部会进一步反向到交叉注意力（从编码器获取信息）
             但编码器的参数不会在此被更新（需要单独的反向传播）

        Args:
            grad_logits: (batch, tgt_len, tgt_vocab_size) 损失对 logits 的梯度
        """
        cache = self._cache

        # 输出投影层梯度:
        #   fc_w: einsum 对 batch 和 seq 维度求和
        self.grad_fc_w = np.einsum('bij,bik->jk', cache['dec_out'], grad_logits)
        self.grad_fc_b = grad_logits.sum(axis=(0, 1))

        # 回传到解码器的梯度: grad_logits @ fc_w^T
        grad_dec = grad_logits @ self.fc_w.T

        # 解码器内部反向传播
        self.decoder.backward(grad_dec)

    def update(self, lr, max_grad_norm=1.0):
        """
        参数更新：收集所有梯度 → 全局梯度裁剪 → SGD 更新。

        步骤：
          1. 收集所有参数梯度（投影层 + 编码器 + 解码器）
          2. 计算全局 L2 范数
          3. 如果范数超过阈值，按比例缩放所有梯度
          4. 对每个参数执行 SGD 更新

        梯度裁剪的作用：
          - 防止梯度爆炸（gradient explosion）
          - 特别是在从零开始训练、没有预训练权重时尤为重要
          - 裁剪不改变梯度方向，只限制其大小

        Args:
            lr:            学习率
            max_grad_norm: 梯度裁剪的最大 L2 范数阈值
        """
        grads = []

        # 收集输出投影层的梯度
        for name in ['grad_fc_w', 'grad_fc_b']:
            g = getattr(self, name)
            if isinstance(g, np.ndarray):
                grads.append(g)

        # 递归收集编码器和解码器中所有子模块的梯度
        for mod in [self.encoder, self.decoder]:
            _collect_grads(mod, grads)

        # 计算全局梯度范数
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))

        # 如果范数超过阈值，缩放所有梯度
        if total_norm > max_grad_norm:
            scale = max_grad_norm / (total_norm + 1e-12)
            for g in grads:
                g *= scale

        # SGD 参数更新
        self.fc_w -= lr * self.grad_fc_w
        self.fc_b -= lr * self.grad_fc_b
        self.encoder.update(lr)
        self.decoder.update(lr)

    def compute_loss(self, src, tgt, src_mask, tgt_mask, target, loss='cross_entropy'):
        """
        一步完整的训练迭代：前向传播 → 计算损失 → 反向传播。

        这是最常用的训练接口，在一次调用中完成：
          1. 前向传播得到 logits
          2. Softmax 计算概率分布
          3. 交叉熵损失计算
          4. 反向传播计算所有梯度
          （注意：不包含参数更新，需要单独调用 update()）

        数值稳定性技巧：
          softmax 计算前减去 logits 的最大值（log-sum-exp trick），
          防止 exp(large_number) 导致溢出。

        交叉熵梯度（闭式解）：
          对于 softmax + 交叉熵，梯度等于概率减去 one-hot 标签，
          即 grad[i] = p[i] 当 i ≠ target，grad[target] = p[target] - 1

        Args:
            src:      (batch, tgt_len) 源序列 token ID
            tgt:      (batch, tgt_len) 目标序列 token ID
            src_mask: 源序列 padding mask
            tgt_mask: 目标序列组合 mask
            target:   (batch, tgt_len) 真实的目标 token ID
            loss:     损失类型（目前仅支持 cross_entropy）

        Returns:
            loss_value: 标量交叉熵损失值
        """
        # 步骤 1: 前向传播
        logits = self.forward(src, tgt, src_mask, tgt_mask)

        # 步骤 2: Softmax 计算概率（使用 log-sum-exp trick 保证数值稳定）
        # 减去最大值防止 exp 溢出
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # 步骤 3: 计算交叉熵损失
        batch_size, tgt_len, vocab_size = probs.shape
        target_flat = target.reshape(-1)          # (batch * tgt_len,)
        probs_2d = probs.reshape(-1, vocab_size)  # (batch * tgt_len, vocab_size)
        N = batch_size * tgt_len

        # 提取每个位置正确 token 的预测概率
        correct_probs = probs_2d[np.arange(N), target_flat]

        # 交叉熵: -mean(log(p_correct))
        # 加 1e-12 防止 log(0) 导致 -Inf
        loss_value = -np.mean(np.log(correct_probs + 1e-12))

        # 步骤 4: 计算梯度（softmax + cross_entropy 的闭式解）
        grad = probs.copy()
        grad_2d = grad.reshape(-1, vocab_size)
        # 对正确 token 位置: p[target] - 1
        grad_2d[np.arange(N), target_flat] -= 1.0
        # 除以 N 取平均
        grad_2d /= N
        grad = grad_2d.reshape(batch_size, tgt_len, vocab_size)

        # 步骤 5: 反向传播
        self.backward(grad)

        return loss_value

    def save(self, path):
        """
        保存模型权重到 npz 文件。

        注意：当前实现仅保存输出投影层，完整实现应递归保存
        编码器和解码器的所有参数。

        Args:
            path: 保存路径（不含扩展名，会生成 .npz 和 .npz.json）
        """
        import json

        # 保存投影层权重
        params = {'fc_w': self.fc_w, 'fc_b': self.fc_b}
        np.savez(path, **params)

        # 保存模型配置
        config = {'d_model': self.d_model}
        with open(path + '.json', 'w') as f:
            json.dump(config, f)

    def predict(self, src, src_mask, start_idx, end_idx, max_len=50):
        """
        自回归预测（Greedy Decoding / 贪婪解码）。

        用于推理阶段，逐个生成目标 token：
          1. 以 start_idx 作为第一个 token
          2. 用解码器预测下一个 token 的 argmax
          3. 将生成的 token 追加到目标序列
          4. 重复直到生成 end_idx 或达到 max_len

        与训练时的区别：
          - 训练: 整个目标序列同时输入（teacher forcing）
          - 推理: 逐个 token 生成（自回归）

        Args:
            src:       (1, src_len) 单个样本的源序列
            src_mask:  源序列 mask
            start_idx: 序列起始 token（如 <SOS>）
            end_idx:   序列结束 token（如 <EOS>）
            max_len:   最大生成长度

        Returns:
            generated: 生成的 token ID 列表
        """
        # 步骤 1: 编码源序列（只需编码一次）
        enc_out = self.encoder.forward(src, src_mask)

        # 步骤 2: 从 start_idx 开始自回归生成
        generated = [start_idx]

        for _ in range(max_len - 1):
            # 将已生成的 token 转为张量
            tgt = np.array([generated])
            tgt_len = len(generated)

            # 创建因果 mask（生成序列长度变化，需要每次重建）
            tgt_mask = np.tril(
                np.ones((1, 1, tgt_len, tgt_len), dtype=np.float32)
            )

            # 解码器前向
            dec_out = self.decoder.forward(tgt, enc_out, src_mask, tgt_mask)

            # 线性投影
            logits = dec_out @ self.fc_w + self.fc_b

            # 取最后一个位置的预测作为下一个 token（贪婪选择）
            next_token = np.argmax(logits[0, -1, :])
            generated.append(int(next_token))

            # 如果生成了结束 token，停止生成
            if next_token == end_idx:
                break

        return generated
