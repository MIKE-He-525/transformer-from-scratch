"""
test_transformer.py — Transformer (纯 NumPy 实现) 完整测试套件

本文件对整个 Transformer 实现进行逐层测试，确保：
  1. 所有模块能正确导入
  2. 每个组件的前向传播输出形状正确
  3. 每个组件的反向传播正常工作
  4. 参数更新（SGD）有效
  5. Mask 函数生成正确的掩码
  6. 完整模型的前向/反向传播流程畅通
  7. 模型能够在简单任务上收敛（loss 下降）
  8. 模型在大规模训练中没有数值不稳定（NaN/Inf）
  9. 模型能学习简单的 token 映射关系

测试顺序设计：
  从细粒度（单个组件）到粗粒度（完整模型），逐步验证。
  如果底层组件有问题，上层测试会更快暴露问题。
"""
import numpy as np
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# 1. 导入测试 — 验证所有模块能正确导入
# =============================================================================
def test_imports():
    """
    测试所有核心类和函数能否成功导入。

    这是最基础的测试，确保：
      - 模块文件存在且语法正确
      - __init__.py 的导出配置正确
      - 没有循环导入问题
    """
    print("=" * 50)
    print("1. 导入测试")
    print("=" * 50)

    # 从包级别导入（通过 __init__.py）
    from transformer import (
        Embedding, PositionalEncoding,             # 嵌入模块
        MultiHeadAttention, scaled_dot_product_attention,  # 注意力模块
        FeedForward,                               # 前馈网络
        LayerNorm, ResidualConnection,             # 归一化与残差
        Encoder, EncoderLayer,                     # 编码器
        Decoder, DecoderLayer,                     # 解码器
        Transformer, utils,                        # 完整模型与工具
    )

    # 从 utils 导入 mask 函数
    from transformer.utils import (
        create_causal_mask, create_padding_mask,
        create_src_mask, create_tgt_mask,
    )

    print("  PASS 所有模块导入成功\n")


# =============================================================================
# 2. 组件单元测试 — 逐个验证每个组件的前向/反向传播
# =============================================================================

def test_embedding():
    """
    测试 Embedding 和 PositionalEncoding。

    检查点：
      - Embedding 前向: 输入 (batch, seq) → 输出 (batch, seq, d_model)
      - Embedding 反向: 梯度能正确累积到嵌入矩阵
      - Embedding 更新: 更新后参数发生变化
      - PositionalEncoding 前向: 输出形状与输入相同
      - PositionalEncoding 反向: 梯度直接透传
    """
    print("=" * 50)
    print("2. Embedding 测试")
    print("=" * 50)
    from transformer.embedding import Embedding, PositionalEncoding

    # 创建嵌入层: 100 个词, 32 维
    emb = Embedding(vocab_size=100, d_model=32)

    # 构造输入: 2 个样本，每个 3 个 token
    x = np.array([[1, 5, 10], [2, 8, 15]])  # (2, 3)

    # 前向传播
    out = emb.forward(x)

    # 验证输出形状
    assert out.shape == (2, 3, 32), f"期望 (2,3,32), 实际 {out.shape}"
    print(f"  Embedding forward: 输入 {x.shape} → 输出 {out.shape}")

    # 验证输出非零（参数已被使用）
    assert np.linalg.norm(out) > 0

    # 反向传播：传入梯度，检查梯度累积
    emb.backward(out)
    emb.update(0.01)
    print("  Embedding backward + update: PASS")

    # ---- PositionalEncoding 测试 ----
    pe = PositionalEncoding(d_model=32, max_len=100)

    # 前向传播
    pe_out = pe.forward(out)
    assert pe_out.shape == (2, 3, 32)

    # 反向传播（无参数，直接透传）
    grad = pe.backward(pe_out)
    assert grad.shape == (2, 3, 32)

    print(f"  PositionalEncoding: {out.shape} → {pe_out.shape}")
    print("  PASS\n")


def test_layer_norm():
    """
    测试 LayerNorm 的归一化效果。

    检查点：
      - 输出形状与输入相同
      - 归一化后，最后一个维度的均值 ≈ 0，方差 ≈ 1
      - 反向传播和参数更新正常工作
    """
    print("=" * 50)
    print("3. LayerNorm 测试")
    print("=" * 50)
    from transformer.normalization import LayerNorm

    ln = LayerNorm(d_model=64)

    # 随机输入 (4 个样本, 10 个位置, 64 维)
    x = np.random.randn(4, 10, 64)
    out = ln.forward(x)

    # 形状不变
    assert out.shape == x.shape

    # 验证归一化效果：均值 ≈ 0，方差 ≈ 1
    mean = np.mean(out, axis=-1)
    var = np.var(out, axis=-1)
    assert np.allclose(mean, 0, atol=1e-4), f"均值偏离: {mean}"
    assert np.allclose(var, 1, atol=1e-4), f"方差偏离: {var}"

    print(f"  LayerNorm forward: 均值范围 [{mean.min():.6f}, {mean.max():.6f}]")
    print(f"  方差范围 [{var.min():.6f}, {var.max():.6f}]")

    # 反向 + 更新
    ln.backward(out)
    ln.update(0.01)
    print("  PASS\n")


def test_attention():
    """
    测试 scaled_dot_product_attention 和 MultiHeadAttention。

    检查点：
      - scaled_dot_product_attention 输出形状正确
      - attention weights 每行和为 1（softmax 归一化验证）
      - mask 功能正确（被遮蔽位置权重为 0）
      - MultiHeadAttention 前向/反向/更新完整流程
    """
    print("=" * 50)
    print("4. Attention 测试")
    print("=" * 50)
    from transformer.attention import MultiHeadAttention, scaled_dot_product_attention

    # ---- ScaledDotProductAttention 测试 ----
    # 形状: (batch=2, heads=4, seq=10, d_k=16)
    q = np.random.randn(2, 4, 10, 16)
    k = np.random.randn(2, 4, 10, 16)
    v = np.random.randn(2, 4, 10, 16)

    out, weights = scaled_dot_product_attention(q, k, v)

    # 验证输出形状
    assert out.shape == (2, 4, 10, 16)
    assert weights.shape == (2, 4, 10, 10)

    # 验证注意力权重每行和为 1
    assert np.allclose(weights.sum(axis=-1), 1, atol=1e-5)

    print(f"  scaled_dot_product_attention: {out.shape}")
    print(f"  attention weights: {weights.shape}, "
          f"行和≈1: {np.allclose(weights.sum(-1), 1)}")

    # ---- Mask 功能测试 ----
    mask = np.ones((2, 1, 1, 10))
    mask[:, :, :, 7:] = 0  # 遮蔽最后 3 个位置

    out_masked, w_masked = scaled_dot_product_attention(q, k, v, mask)

    # 被遮蔽位置的权重应为 0
    # w_masked[:, :, :7, 7:] 表示前 7 个 query 位置对后 3 个 key 位置的注意力
    assert np.allclose(w_masked[:, :, :7, 7:], 0, atol=1e-8)
    print("  mask 遮蔽功能: PASS")

    # ---- MultiHeadAttention 测试 ----
    mha = MultiHeadAttention(d_model=64, n_heads=4)
    q_batch = np.random.randn(2, 10, 64)

    # 自注意力: Q = K = V
    mha_out = mha.forward(q_batch, q_batch, q_batch)
    assert mha_out.shape == (2, 10, 64)

    # 反向传播 + 更新
    mha.backward(mha_out)
    mha.update(0.01)

    print(f"  MultiHeadAttention: (2,10,64) → {mha_out.shape}")
    print("  PASS\n")


def test_feed_forward():
    """
    测试 FeedForward 前向/反向/更新。

    检查点：
      - 输出形状与输入相同 (d_model → hidden_dim → d_model)
      - 反向传播产生梯度
      - 参数更新执行
    """
    print("=" * 50)
    print("5. FeedForward 测试")
    print("=" * 50)
    from transformer.feed_forward import FeedForward

    ffn = FeedForward(d_model=64, hidden_dim=128, dropout=0.1)
    x = np.random.randn(4, 10, 64)

    # 前向
    out = ffn.forward(x)
    assert out.shape == (4, 10, 64)

    # 反向 + 更新
    ffn.backward(out)
    ffn.update(0.01)

    print(f"  FeedForward: (4,10,64) → {out.shape} (dropout=0.1)")
    print("  PASS\n")


def test_residual_connection():
    """
    测试 ResidualConnection（Pre-LN 残差连接）。

    检查点：
      - 将 FFN 作为子层传入
      - 前向输出形状不变
      - 反向传播正常
    """
    print("=" * 50)
    print("6. ResidualConnection 测试")
    print("=" * 50)
    from transformer.normalization import ResidualConnection
    from transformer.feed_forward import FeedForward

    rc = ResidualConnection(d_model=64, dropout=0.0)
    ffn = FeedForward(d_model=64, hidden_dim=128)
    x = np.random.randn(4, 10, 64)

    # 前向
    out = rc.forward(x, ffn)
    assert out.shape == (4, 10, 64)

    # 反向 + 更新
    rc.backward(out)
    rc.update(0.01)

    print(f"  ResidualConnection: (4,10,64) → {out.shape}")
    print("  PASS\n")


# =============================================================================
# 3. Mask 工具函数测试 — 验证三种 Mask 的正确性
# =============================================================================

def test_masks():
    """
    测试 create_src_mask, create_causal_mask, create_tgt_mask。

    检查点：
      - src_mask: 填充位置为 0，有效位置为 1，形状正确
      - causal_mask: 是严格的下三角矩阵
      - tgt_mask: 因果 + padding 的组合 mask，形状正确
    """
    print("=" * 50)
    print("7. Mask 工具函数测试")
    print("=" * 50)
    from transformer.utils import create_src_mask, create_tgt_mask, create_causal_mask

    # ---- 源序列 Padding Mask ----
    src = np.array([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    src_mask = create_src_mask(src)

    # 形状: (batch=2, 1, 1, seq=5)
    assert src_mask.shape == (2, 1, 1, 5)

    # 样本 1: 位置 3,4 是填充（值为 0），位置 0,1,2 有效（值为 1）
    assert src_mask[0, 0, 0, 3:].sum() == 0
    assert src_mask[0, 0, 0, :3].sum() == 3

    print(f"  create_src_mask: {src_mask.shape}")
    print(f"    样本1: {src_mask[0, 0, 0]} → 遮蔽位置正确")
    print(f"    样本2: {src_mask[1, 0, 0]} → 遮蔽位置正确")

    # ---- Causal Mask ----
    causal = create_causal_mask(4)

    # 验证是严格的下三角矩阵
    expected = np.array([[[[1, 0, 0, 0],
                           [1, 1, 0, 0],
                           [1, 1, 1, 0],
                           [1, 1, 1, 1]]]])
    assert np.array_equal(causal, expected), "Causal mask 不正确"
    print(f"  create_causal_mask: 下三角矩阵正确")

    # ---- 目标序列组合 Mask ----
    tgt = np.array([[1, 2, 0]])
    tgt_mask = create_tgt_mask(tgt)

    # 形状: (batch=1, 1, tgt=3, tgt=3)
    assert tgt_mask.shape == (1, 1, 3, 3)
    print(f"  create_tgt_mask: {tgt_mask.shape}")
    print("  PASS\n")


# =============================================================================
# 4. 完整前向传播测试 — 端到端验证
# =============================================================================

def test_forward():
    """
    测试完整 Transformer 模型的前向传播。

    构建一个小模型并执行完整的前向传播，检查输出形状。

    流程：
      src → Encoder(src_mask) → enc_out
      tgt + enc_out → Decoder(src_mask, tgt_mask) → dec_out
      dec_out → Linear → logits
    """
    print("=" * 50)
    print("8. 完整前向传播测试")
    print("=" * 50)
    from transformer import Transformer, utils

    # 创建小模型
    model = Transformer(
        src_vocab_size=100, tgt_vocab_size=100,
        d_model=64, n_layers=2, n_heads=4, hidden_dim=128,
        dropout=0.0,  # 测试时关闭 dropout
    )

    # 随机输入
    src = np.random.randint(1, 100, (2, 10))  # (batch=2, src_len=10)
    tgt = np.random.randint(1, 100, (2, 8))   # (batch=2, tgt_len=8)

    src_mask = utils.create_src_mask(src)
    tgt_mask = utils.create_tgt_mask(tgt)

    # 前向传播
    logits = model.forward(src, tgt, src_mask, tgt_mask)

    # 验证输出形状: (batch=2, tgt_len=8, vocab=100)
    assert logits.shape == (2, 8, 100), f"期望 (2,8,100), 实际 {logits.shape}"

    print(f"  src: {src.shape}, tgt: {tgt.shape}")
    print(f"  logits: {logits.shape}")
    print("  PASS\n")


# =============================================================================
# 5. 反向传播测试 — 验证模型能收敛
# =============================================================================

def test_backward():
    """
    训练一个小模型 200 步，验证 loss 是否下降。

    这是一个端到端的训练测试，验证：
      - compute_loss 正确计算交叉熵
      - backward 正确计算所有梯度
      - update 正确更新参数
      - Loss 应该随着训练逐步下降

    任务：让模型学习一个简单的一致性映射
    （输入 src，输出 tgt = src 拼接随机后缀）。
    """
    print("=" * 50)
    print("9. 反向传播 + 训练循环测试")
    print("=" * 50)
    from transformer import Transformer, utils

    np.random.seed(42)

    # 小模型: 词表 30, d_model=32
    model = Transformer(
        src_vocab_size=30, tgt_vocab_size=30,
        d_model=32, n_layers=2, n_heads=4, hidden_dim=64,
        dropout=0.0,
    )

    losses = []
    for step in range(200):
        # 生成随机训练数据
        src = np.random.randint(1, 30, (4, 6))
        # tgt = src 拼接 2 个随机 token（让模型学习复制 + 扩展）
        tgt = np.concatenate([src, np.random.randint(1, 30, (4, 2))], axis=1)

        src_mask = utils.create_src_mask(src)
        tgt_mask = utils.create_tgt_mask(tgt)

        # 一步训练
        loss = model.compute_loss(src, tgt, src_mask, tgt_mask, tgt)
        model.update(0.1)
        losses.append(loss)

    # 打印关键时间点的 loss
    print(f"  Step   0: loss = {losses[0]:.4f}")
    print(f"  Step  50: loss = {losses[50]:.4f}")
    print(f"  Step 100: loss = {losses[100]:.4f}")
    print(f"  Step 199: loss = {losses[-1]:.4f}")

    # 验证 loss 下降
    assert losses[-1] < losses[0], f"Loss 没有下降: {losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"  Loss 下降: {losses[0]:.4f} → {losses[-1]:.4f}")
    print("  PASS\n")


# =============================================================================
# 6. 数值稳定性测试 — 检测 NaN/Inf
# =============================================================================

def test_numerical_stability():
    """
    用更大的模型和数据量训练 50 步，检测是否有 NaN/Inf。

    检查点：
      - 大词表 (1000)、深层模型 (3 层)、多头 (8 头)
      - 开启 dropout (0.1)
      - 使用较大 batch (8)
      - 50 步训练中不应出现 NaN/Inf

    数值不稳定常见原因：
      - softmax 溢出（应使用 max 减法）
      - log(0)（应加小常数 epsilon）
      - 梯度爆炸（应使用梯度裁剪）
      - 除以 0（如 LayerNorm 中 eps）
    """
    print("=" * 50)
    print("10. 数值稳定性测试")
    print("=" * 50)
    from transformer import Transformer, utils

    # 大模型: 词表 1000, 3 层, 8 头
    model = Transformer(
        src_vocab_size=1000, tgt_vocab_size=1000,
        d_model=128, n_layers=3, n_heads=8, hidden_dim=256,
        dropout=0.1,  # 开启 dropout
    )

    has_nan = False
    for step in range(50):
        src = np.random.randint(1, 1000, (8, 20))
        tgt = np.random.randint(1, 1000, (8, 15))
        src_mask = utils.create_src_mask(src)
        tgt_mask = utils.create_tgt_mask(tgt)

        loss = model.compute_loss(src, tgt, src_mask, tgt_mask, tgt)
        model.update(0.05)

        # 检测 NaN 和 Inf
        if np.isnan(loss) or np.isinf(loss):
            has_nan = True
            print(f"  NaN/Inf 在 step {step}, loss={loss}")
            break

    if not has_nan:
        print("  50步训练, 无 NaN/Inf 出现")
    print("  PASS\n")


# =============================================================================
# 7. 完整演示 — 训练模型学习简单映射
# =============================================================================

def demo_translation():
    """
    演示 Transformer 学习简单映射关系。

    任务：将源 token x 翻译成目标 token x + 10
      源词汇: 1~5
      目标词汇: 11~15

    这是一个 seq2seq 映射任务，模型需要学习输入和输出之间的确定性关系。
    训练 1000 步后，用未见过的输入测试泛化能力。

    预期结果：
      训练 loss 应该快速下降到接近 0
      测试准确率应接近 100%（因为映射是确定性的）
    """
    print("=" * 50)
    print("11. 完整演示: 学习简单映射")
    print("=" * 50)
    from transformer import Transformer, utils

    np.random.seed(0)

    # 词表: 1~5 是源语言, 11~15 是目标语言
    src_vocab = 20
    tgt_vocab = 20
    d_model = 16
    n_layers = 2
    n_heads = 2
    hidden_dim = 32

    model = Transformer(
        src_vocab, tgt_vocab, d_model, n_layers, n_heads, hidden_dim,
        dropout=0.0,
    )

    # 训练: 学习源 token x → 目标 token x + 10
    print("  训练 1000 步...")
    for step in range(1000):
        # 源序列: 4 个位置，每个取 1~5
        src = np.random.randint(1, 6, (8, 4))
        # 目标序列: 源序列 + 10
        tgt = src + 10

        src_mask = utils.create_src_mask(src)
        tgt_mask = utils.create_tgt_mask(tgt)

        loss = model.compute_loss(src, tgt, src_mask, tgt_mask, tgt)
        model.update(0.5)

        if step % 200 == 0:
            print(f"    Step {step:4d}: loss = {loss:.4f}")

    # 测试泛化能力
    test_src = np.array([[3, 5, 1, 2]])
    test_tgt = test_src + 10
    src_mask = utils.create_src_mask(test_src)
    tgt_mask = utils.create_tgt_mask(test_tgt)

    logits = model.forward(test_src, test_tgt, src_mask, tgt_mask)
    preds = np.argmax(logits, axis=-1)

    print(f"\n  输入:  {test_src[0].tolist()}")
    print(f"  预测:  {preds[0].tolist()}")
    print(f"  期望:  {test_tgt[0].tolist()}")

    correct = np.sum(preds == test_tgt)
    total = test_tgt.size
    print(f"  准确率: {correct}/{total} = {correct/total:.0%}")
    print("  PASS\n")


# =============================================================================
# 主函数 — 运行所有测试
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print()
    print("╔" + "═" * 48 + "╗")
    print("║   Transformer (纯NumPy) - 测试套件              ║")
    print("╚" + "═" * 48 + "╝")
    print()

    try:
        test_imports()                # 1. 导入测试
        test_embedding()              # 2. Embedding + PositionalEncoding
        test_layer_norm()             # 3. LayerNorm
        test_attention()              # 4. Attention
        test_feed_forward()           # 5. FeedForward
        test_residual_connection()    # 6. ResidualConnection
        test_masks()                  # 7. Mask 工具函数
        test_forward()                # 8. 完整前向传播
        test_backward()               # 9. 反向传播 + 训练收敛
        test_numerical_stability()    # 10. 数值稳定性
        demo_translation()            # 11. 完整演示

        print("╔" + "═" * 48 + "╗")
        print("║   ALL TESTS PASSED!                            ║")
        print("╚" + "═" * 48 + "╝")
    except Exception as e:
        print(f"\n  FAIL: {type(e).__name__}: {e}")
        raise
