"""
flight_price_predictor.py — 航班价格预测（使用纯 NumPy Transformer Encoder）

本项目使用从零实现的 Transformer Encoder 架构，对航班价格进行回归预测。
与 Seq2Seq 翻译任务不同，这里是 Encoder-only 架构：将航班特征编码后，
通过一个线性层直接回归出价格。

== 模型架构（Encoder-only 回归）==

    航班特征 (stops, class, duration, ...)
         │
         ▼
    ┌──────────────────┐
    │ Continuous       │   连续特征线性投影到 d_model 维
    │ Embedding        │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Positional       │   正弦位置编码
    │ Encoding         │
    └────────┬─────────┘
             │
             ▼
    ╔════════════════════╗
    ║ Encoder Layer × N  ║   N 层 Transformer Encoder
    ║ (Self-Attn + FFN)  ║
    ╚════════════════════╝
             │
             ▼
    ┌──────────────────┐
    │ LayerNorm        │   最终归一化
    │ Linear(d→1)      │   回归到单个价格值
    └──────────────────┘
             │
             ▼
        预测价格（标量）

== 数据集说明 ==

    数据集包含三种航线（route）的航班信息：
      - bangalore_delhi: 班加罗尔 → 德里
      - delhi_mumbai: 德里 → 孟买
      - mumbai_delhi: 孟买 → 德里

    每条航线有 train/test CSV，特征包括：
      - stops: 经停次数（0/1/2）
      - class: 舱位类型（0=经济舱, 1=商务舱）
      - duration: 飞行时长（归一化后）
      - days_left: 距起飞天数（归一化后）
      - airline_*: 航空公司独热编码
      - departure_time_*: 出发时段独热编码
      - arrival_time_*: 到达时段独热编码
      - route_*: 航线独热编码

== 多航线联合训练 ==

    所有航线的训练数据合并后统一训练一个模型。
    每条航线的数据附带 route_* 独热编码，使模型能区分不同航线。
    这样可以利用更多数据学习更通用的价格模式。

== 可视化分析 ==

    训练过程中生成 7 种图表：
      01: 数据分布（价格直方图、天数-价格散点等）
      02: 特征重要性（基于相关性和均值差异）
      03: 训练曲线（RMSE，线性 & 对数坐标）
      04: days_left 敏感性分析
      05: 预订时间线（最优购票时机）
      06: 特征敏感性（每个特征对价格的影响）
      07: 预测 vs 实际散点图（含 RMSE, R², MAE）
"""
import numpy as np
import pandas as pd

# 使用非交互后端（无 GUI 环境下保存图片）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置中文字体，确保图表中的中文能正确显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
# 修复负号显示
plt.rcParams['axes.unicode_minus'] = False

import os
import sys

# 确保可以导入 transformer 子模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# 1. 连续特征 Transformer 回归模型
# =============================================================================

class ContinuousEmbedding:
    """
    连续特征线性嵌入层。

    作用：将原始特征向量（如 stops, duration, days_left 等）线性投影到
    Transformer 所需的 d_model 维空间。

    与词嵌入的区别：
      - 词嵌入: 离散索引 → 查表 → 向量
      - 连续嵌入: 连续值 → 线性变换 → 向量

    初始化使用 He 风格的 Xavier 均匀分布：
      scale = sqrt(2 / (input_dim + d_model))
    这使输入和输出梯度的方差保持均衡。

    反向传播：
      forward: y = x @ W + b
      backward:
        - dW = Σ x^T · grad  (对 batch 维度求和)
        - db = Σ grad        (对 batch 维度求和)
        - dx = grad @ W^T    (回传)
    """

    def __init__(self, input_dim, d_model):
        """
        初始化连续特征嵌入层。

        Args:
            input_dim:  原始特征维度（即特征列数）
            d_model:    投影后的 Transformer 模型维度
        """
        # Xavier 均匀分布初始化范围
        scale = np.sqrt(2.0 / (input_dim + d_model))
        self.weight = np.random.uniform(-scale, scale, (input_dim, d_model))
        self.bias = np.zeros(d_model)

        # 梯度数组
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        # 缓存前向传播的输入
        self._x = None

    def forward(self, x):
        """
        前向传播：线性投影。

        Args:
            x: (batch, seq, input_dim) 连续特征输入

        Returns:
            (batch, seq, d_model) 投影后的特征向量
        """
        self._x = x
        return x @ self.weight + self.bias

    def backward(self, grad):
        """
        反向传播：计算权重和偏置的梯度，回传输入梯度。

        Args:
            grad: (batch, seq, d_model) 来自上层的梯度

        Returns:
            (batch, seq, input_dim) 回传到输入的梯度
        """
        x = self._x

        # 权重梯度: Σ over batch 和 seq 维度
        self.grad_weight = np.einsum('bij,bik->jk', x, grad)

        # 偏置梯度: Σ over batch 和 seq 维度
        self.grad_bias = grad.sum(axis=(0, 1))

        # 回传到输入
        return grad @ self.weight.T

    def update(self, lr):
        """
        SGD 参数更新。

        Args:
            lr: 学习率
        """
        self.weight -= lr * self.grad_weight
        self.bias -= lr * self.grad_bias


class TransformerEncoder:
    """
    纯 Encoder Transformer 回归模型。

    本模型用于回归任务：接收一组连续特征，输出一个标量价格预测。
    与标准 Seq2Seq Transformer 不同，这里不需要解码器，只需要编码器。

    架构流程：
      连续特征 → ContinuousEmbedding → PositionalEncoding →
      [Multi-Head Self-Attention + Residual] × N →
      [FeedForward + Residual] × N →
      LayerNorm → Linear(d_model → 1) → 预测价格

    训练过程：
      1. 每个样本 reshape 为 (batch, 1, input_dim)，即序列长度为 1
      2. 经过编码器后得到 (batch, 1, d_model) 的隐状态
      3. 线性层映射为 (batch, 1, 1) 的预测价格
      4. 与真实价格计算 MSE 损失
      5. 反向传播 + 梯度裁剪 + SGD 更新

    梯度裁剪（Gradient Clipping）：
      收集所有子模块的梯度张量，计算全局 L2 范数，
      若超过阈值则按比例缩放，防止梯度爆炸。
    """

    def __init__(self, input_dim, d_model=64, n_layers=2, n_heads=4, hidden_dim=128, dropout=0.1):
        """
        初始化 Transformer Encoder 回归模型。

        Args:
            input_dim:  输入特征维度
            d_model:    模型维度
            n_layers:   Encoder 层数
            n_heads:    注意力头数
            hidden_dim: FFN 隐藏层维度
            dropout:    Dropout 比率
        """
        # 延迟导入，避免循环依赖
        from transformer.attention import MultiHeadAttention
        from transformer.feed_forward import FeedForward
        from transformer.normalization import ResidualConnection, LayerNorm
        from transformer.embedding import PositionalEncoding

        self.d_model = d_model

        # 连续特征嵌入层
        self.input_proj = ContinuousEmbedding(input_dim, d_model)

        # 正弦位置编码
        self.pos_enc = PositionalEncoding(d_model)

        # 堆叠 N 个编码器层
        # 每层包含: 自注意力 + FFN + 两个残差连接
        self.layers = []
        for _ in range(n_layers):
            self_attn = MultiHeadAttention(d_model, n_heads)
            ffn = FeedForward(d_model, hidden_dim, dropout)
            res1 = ResidualConnection(d_model, dropout)  # 自注意力残差
            res2 = ResidualConnection(d_model, dropout)  # FFN 残差
            self.layers.append({
                'self_attn': self_attn,
                'ffn': ffn,
                'res1': res1,
                'res2': res2,
            })

        # 最终归一化层
        self.final_norm = LayerNorm(d_model)

        # 输出线性层: d_model → 1（回归到单个价格值）
        self.out = _Linear(d_model, 1)

    def forward(self, x, mask=None):
        """
        前向传播。

        Args:
            x:     (batch, seq, input_dim) 连续特征
            mask:  可选，mask 参数（回归任务通常不使用）

        Returns:
            (batch, seq, 1) 预测价格
        """
        # 步骤 1: 连续特征嵌入
        x = self.input_proj.forward(x)

        # 步骤 2: 位置编码
        x = self.pos_enc.forward(x)

        # 步骤 3: 通过所有编码器层
        for layer in self.layers:
            # 自注意力子层
            x = layer['res1'].forward(x, layer['self_attn'], x, x, x, mask)
            # FFN 子层
            x = layer['res2'].forward(x, layer['ffn'])

        # 步骤 4: 最终归一化
        x = self.final_norm.forward(x)

        # 步骤 5: 回归输出
        return self.out.forward(x)

    def backward(self, grad):
        """
        反向传播。

        按照前向传播的完全逆序：
          输出线性层 → 最终归一化 → 编码器层 N → ... → 编码器层 1
          → 位置编码 → 连续特征嵌入

        Args:
            grad: (batch, seq, 1) 来自损失函数的梯度
        """
        # 输出层反向
        grad = self.out.backward(grad)

        # 归一化层反向
        grad = self.final_norm.backward(grad)

        # 编码器层逆序反向
        for layer in reversed(self.layers):
            grad = layer['res2'].backward(grad)  # FFN 残差
            grad = layer['res1'].backward(grad)  # 自注意力残差

        # 位置编码反向（无参数，直接透传）
        # 连续特征嵌入反向
        self.input_proj.backward(self.pos_enc.backward(grad))

    def update(self, lr, max_grad_norm=1.0):
        """
        参数更新（含梯度裁剪）。

        流程：
          1. 收集所有子模块的梯度
          2. 计算全局梯度范数
          3. 如超过阈值则缩放
          4. SGD 更新所有参数

        Args:
            lr:            学习率
            max_grad_norm: 梯度裁剪的最大范数
        """
        # 收集输出层的梯度
        grads = [
            g for name in dir(self.out)
            if name.startswith('grad_')
            for g in [getattr(self.out, name)]
            if isinstance(g, np.ndarray)
        ]

        # 从 transformer 模型中导入梯度收集函数
        from transformer.model import _collect_grads as _cg

        # 收集所有编码器层的梯度
        for layer in self.layers:
            _cg(layer['self_attn'], grads)   # 自注意力梯度
            _cg(layer['ffn'], grads)         # FFN 梯度
            _cg(layer['res1'], grads)        # 残差连接 1 的梯度 (LayerNorm)
            _cg(layer['res2'], grads)        # 残差连接 2 的梯度 (LayerNorm)

        # 收集其他组件的梯度
        _cg(self.final_norm, grads)
        _cg(self.input_proj, grads)

        # 计算全局梯度范数
        norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))

        # 梯度裁剪
        if norm > max_grad_norm:
            for g in grads:
                g *= max_grad_norm / (norm + 1e-12)

        # SGD 更新所有参数
        self.out.update(lr)
        self.input_proj.update(lr)
        self.final_norm.update(lr)
        for layer in self.layers:
            layer['self_attn'].update(lr)
            layer['ffn'].update(lr)
            layer['res1'].update(lr)
            layer['res2'].update(lr)


class _Linear:
    """
    简单线性层（无框架实现）。

    用于将 d_model 维的编码器输出映射为 1 维的价格预测。

    forward:  y = x @ W + b
    backward: dW = x^T · grad, db = Σ grad, dx = grad @ W^T
    """

    def __init__(self, in_f, out_f):
        """
        初始化线性层。

        Args:
            in_f:  输入特征维度
            out_f: 输出特征维度
        """
        # Xavier 均匀分布初始化
        scale = np.sqrt(2.0 / (in_f + out_f))
        self.w = np.random.uniform(-scale, scale, (in_f, out_f))
        self.b = np.zeros(out_f)

        # 梯度数组
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

        # 缓存输入
        self._x = None

    def forward(self, x):
        """
        前向传播。

        Args:
            x: (batch, seq, in_f) 输入

        Returns:
            (batch, seq, out_f) 输出
        """
        self._x = x
        return x @ self.w + self.b

    def backward(self, grad):
        """
        反向传播。

        Args:
            grad: (batch, seq, out_f) 上游梯度

        Returns:
            (batch, seq, in_f) 回传梯度
        """
        x = self._x

        # 累积权重和偏置梯度
        self.grad_w = np.einsum('bij,bik->jk', x, grad)
        self.grad_b = grad.sum(axis=(0, 1))

        # 回传到输入
        return grad @ self.w.T

    def update(self, lr):
        """
        SGD 参数更新。

        Args:
            lr: 学习率
        """
        self.w -= lr * self.grad_w
        self.b -= lr * self.grad_b


# =============================================================================
# 2. 数据加载与预处理
# =============================================================================

# 模型输入使用的特征列
# 这些特征涵盖了航班的各类信息：
#   - stops: 经停次数（数值型）
#   - class: 舱位类型（数值型，0/1）
#   - duration: 飞行时长（归一化数值型）
#   - days_left: 距起飞天数（归一化数值型）
#   - airline_*: 航空公司独热编码（分类型 → one-hot）
#   - departure_time_*: 出发时段独热编码（分时段 → one-hot）
#   - arrival_time_*: 到达时段独热编码（分时段 → one-hot）
#   - route_*: 航线独热编码（分航线 → one-hot）
FEATURE_COLS = [
    'stops', 'class', 'duration', 'days_left',
    'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo',
    'airline_SpiceJet', 'airline_VistARA',
    'departure_time_Early_Morning', 'departure_time_Evening',
    'departure_time_Late_Night', 'departure_time_Morning',
    'departure_time_Night',
    'arrival_time_Early_Morning', 'arrival_time_Evening',
    'arrival_time_Late_Night', 'arrival_time_Morning',
    'arrival_time_Night',
    'route_bangalore_delhi', 'route_delhi_mumbai', 'route_mumbai_delhi',
]

# 输入特征维度
INPUT_DIM = len(FEATURE_COLS)

# 航线列表
ROUTES = ['bangalore_delhi', 'delhi_mumbai', 'mumbai_delhi']


def load_route(route_name, cabin='economy'):
    """
    加载单条航线的训练集和测试集。

    数据预处理步骤：
      1. 读取 train/test CSV 文件
      2. 将布尔类型列转换为浮点型（0.0/1.0），因为 NumPy 不直接支持布尔运算
      3. 为每条航线添加 route_* 独热编码列
         - 当前航线的 route 列标记为 1
         - 其他航线的 route 列标记为 0

    Args:
        route_name: 航线名称（如 'bangalore_delhi'）
        cabin:      舱位类型（'economy' 或 'business'）

    Returns:
        df_train, df_test: 处理后的 DataFrame
    """
    df_train = pd.read_csv(f'dataset/train_{route_name}_{cabin}.csv')
    df_test = pd.read_csv(f'dataset/test_{route_name}_{cabin}.csv')

    for df in [df_train, df_test]:
        # 布尔列 → 浮点列
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(float)

        # 添加航线独热编码
        for route in ROUTES:
            df['route_' + route] = 1 if route == route_name else 0

    return df_train, df_test


def load_all_routes(cabin='economy'):
    """
    加载并合并所有航线的训练集和测试集。

    将三条航线的训练数据拼接为一个大的训练集，
    测试数据拼接为一个大的测试集。这样可以让模型
    利用更多数据学习通用的价格模式。

    Args:
        cabin: 舱位类型

    Returns:
        df_train_all, df_test_all: 合并后的训练集和测试集
    """
    all_train, all_test = [], []

    for route in ROUTES:
        df_train, df_test = load_route(route, cabin=cabin)
        all_train.append(df_train)
        all_test.append(df_test)
        print(f"  加载 {route} ({cabin}): Train={len(df_train)}, Test={len(df_test)}")

    # 拼接所有航线的训练/测试数据
    df_train_all = pd.concat(all_train, ignore_index=True)
    df_test_all = pd.concat(all_test, ignore_index=True)

    print(f"  合并后总计：Train={len(df_train_all)}, Test={len(df_test_all)}")

    return df_train_all, df_test_all


def make_tensors(df_train, df_test):
    """
    将 DataFrame 转换为 NumPy 张量并进行标准化。

    标准化步骤：
      1. 提取特征矩阵 X 和标签 y
      2. 使用训练集的均值和标准差对 X 进行 Z-score 标准化: (X - mean) / std
      3. 对 y 也进行 Z-score 标准化（模型预测的是标准化后的价格）
      4. 标准差过小的特征设为 1.0，防止除以接近 0 的数

    注意：标准化参数（均值、标准差）仅从训练集计算，
    测试集使用相同的参数进行变换（防止数据泄露）。

    预测后，需要将标准化后的价格逆变换回原始价格：
      original_price = predicted_normalized * y_std + y_mean

    Args:
        df_train: 训练集 DataFrame
        df_test:  测试集 DataFrame

    Returns:
        X_train: (N_train, INPUT_DIM) 标准化后的训练特征
        y_train: (N_train,) 标准化后的训练标签
        X_test:  (N_test, INPUT_DIM) 标准化后的测试特征
        y_test:  (N_test,) 标准化后的测试标签
        ym:      价格均值（用于逆变换）
        ys:      价格标准差（用于逆变换）
    """
    # 提取特征和标签
    X_train = df_train[FEATURE_COLS].values.astype(np.float64)
    y_train = df_train['price'].values.astype(np.float64)
    X_test = df_test[FEATURE_COLS].values.astype(np.float64)
    y_test = df_test['price'].values.astype(np.float64)

    # 计算训练集的统计量
    xm = X_train.mean(0)  # (INPUT_DIM,) 每个特征的均值
    xs = X_train.std(0)   # (INPUT_DIM,) 每个特征的标准差
    xs[xs < 1e-8] = 1.0   # 防止除以 0

    # 计算价格的统计量（用于后续逆变换）
    ym = y_train.mean()
    ys = y_train.std()

    # Z-score 标准化
    X_train = (X_train - xm) / xs
    X_test = (X_test - xm) / xs
    y_train = (y_train - ym) / ys
    y_test = (y_test - ym) / ys

    return X_train, y_train, X_test, y_test, ym, ys


# =============================================================================
# 3. 训练循环
# =============================================================================

def train(model, X_train, y_train, X_test, y_test, epochs=30, lr=0.01, batch_size=32):
    """
    训练 Transformer Encoder 回归模型。

    训练流程（每个 epoch）：
      1. 随机打乱训练数据
      2. 按 batch_size 分批
      3. 每批：
         a. reshape 为 (batch, 1, INPUT_DIM)，序列长度为 1
         b. 前向传播得到预测
         c. 计算 MSE 损失和梯度: grad = 2*(pred - target) / N
         d. 反向传播 + 参数更新
      4. 计算训练集和验证集 RMSE

    MSE 损失梯度推导：
      Loss = mean((pred - target)²)
      dLoss/dpred = 2 * (pred - target) / N
      其中 N = batch_size * seq_len（这里是 batch_size * 1）

    Args:
        model:      TransformerEncoder 模型
        X_train:    训练特征
        y_train:    训练标签
        X_test:     测试特征
        y_test:     测试标签
        epochs:     训练轮数
        lr:         学习率
        batch_size: 每批样本数

    Returns:
        train_losses: 每轮训练损失列表（MSE）
        val_losses:   每轮验证损失列表（MSE）
    """
    N = len(X_train)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # 随机打乱数据
        idx = np.random.permutation(N)
        Xt, yt = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        nb = 0

        # 分批训练
        for i in range(0, N, batch_size):
            # reshape: (batch_size, INPUT_DIM) → (batch_size, 1, INPUT_DIM)
            xb = Xt[i:i + batch_size].reshape(-1, 1, INPUT_DIM)
            yb = yt[i:i + batch_size]
            bs = xb.shape[0]

            # 前向传播
            pred = model.forward(xb)

            # MSE 损失和梯度
            err = pred - yb.reshape(bs, 1, 1)   # 误差
            loss = np.mean(err ** 2)             # MSE
            grad = 2.0 * err / (bs * 1)         # 梯度

            # 反向传播 + 参数更新
            model.backward(grad)
            model.update(lr)

            epoch_loss += loss
            nb += 1

        # 记录平均训练损失
        train_losses.append(epoch_loss / nb)

        # 验证集评估
        xb = X_test.reshape(-1, 1, INPUT_DIM)
        pred = model.forward(xb)
        val_losses.append(np.mean((pred - y_test.reshape(-1, 1, 1)) ** 2))

        # 定期打印
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"train_rmse={np.sqrt(train_losses[-1]):.4f}  "
                  f"val_rmse={np.sqrt(val_losses[-1]):.4f}")

    return train_losses, val_losses


# =============================================================================
# 4. 可视化分析函数
# =============================================================================

def plot_data(df, route, save_dir):
    """
    绘制数据分布分析图（6 个子图）。

    子图内容：
      (0,0) 价格直方图（含中位数和均值标记线）
      (0,1) 距起飞天数 vs 价格的散点图
      (0,2) 经停次数 vs 平均价格柱状图
      (1,0) 航空公司 vs 平均价格水平条形图
      (1,1) 出发时段 vs 平均价格柱状图
      (1,2) 数值特征相关系数热力图

    Args:
        df:       训练集 DataFrame
        route:    航线描述（图表标题用）
        save_dir: 保存目录
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{route} - 数据分布分析', fontsize=16, fontweight='bold')

    # (0,0) 价格分布直方图
    ax = axes[0, 0]
    ax.hist(df['price'], bins=50, edgecolor='steelblue', alpha=0.7, color='#2196F3')
    ax.axvline(df['price'].median(), color='red', linestyle='--', linewidth=2,
               label=f'Median: {df.price.median():.0f}')
    ax.axvline(df['price'].mean(), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {df.price.mean():.0f}')
    ax.set_title('价格分布')
    ax.set_xlabel('Price')
    ax.set_ylabel('Freq')
    ax.legend()

    # (0,1) days_left vs price 散点图
    ax = axes[0, 1]
    ax.scatter(df['days_left'], df['price'], alpha=0.2, s=8, color='#4CAF50')
    ax.set_title('days_left vs price')
    ax.set_xlabel('days_left')
    ax.set_ylabel('Price')

    # (0,2) 经停次数 vs 平均价格
    ax = axes[0, 2]
    df['stops'] = df['stops'].astype(int)
    stops_m = df.groupby('stops')['price'].mean()
    ax.bar(stops_m.index.astype(str), stops_m.values,
           color=['#FF9800', '#2196F3', '#f44336'][:len(stops_m)],
           edgecolor='black', alpha=0.7)
    ax.set_title('Avg Price by Stops')
    ax.set_xlabel('Stops')
    ax.set_ylabel('Avg Price')

    # (1,0) 航空公司 vs 平均价格（水平条形图）
    ax = axes[1, 0]
    airline_cols = [c for c in df.columns if c.startswith('airline_')]
    ap = {col.replace('airline_', ''): df.loc[df[col] == True, 'price'].mean()
          for col in airline_cols if df[col].sum() > 10}
    ax.barh(list(ap.keys()), list(ap.values()), color='#9C27B0', alpha=0.7, edgecolor='black')
    ax.set_title('Avg Price by Airline')
    ax.set_xlabel('Avg Price')

    # (1,1) 出发时段 vs 平均价格
    ax = axes[1, 1]
    dept_cols = [c for c in df.columns if c.startswith('departure_time_')]
    dp = {col.replace('departure_time_', '').replace('_', ' '):
          df.loc[df[col] == True, 'price'].mean()
          for col in dept_cols if df[col].sum() > 10}
    ax.bar(list(dp.keys()), list(dp.values()), color='#E91E63', alpha=0.7, edgecolor='black')
    ax.set_title('Avg Price by Departure Time')
    ax.set_ylabel('Avg Price')
    ax.tick_params(axis='x', rotation=30)

    # (1,2) 数值特征相关系数热力图
    ax = axes[1, 2]
    corr_cols = ['price', 'stops', 'class', 'duration', 'days_left']
    corr = df[corr_cols].corr()
    im = ax.imshow(corr.values, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(['Price', 'Stops', 'Class', 'Dur', 'DaysL'], rotation=45)
    ax.set_yticklabels(['Price', 'Stops', 'Class', 'Dur', 'DaysL'])
    # 在每个格子中写入相关系数值
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_title('Correlation')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/01_data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(df, save_dir):
    """
    绘制特征重要性水平条形图。

    重要性度量方式：
      - 数值特征: |Pearson 相关系数|
      - 分类特征: |该类别平均价格 - 总体平均价格|

    结果归一化到 [0, 1] 范围后绘制。

    Args:
        df:       训练集 DataFrame
        save_dir: 保存目录
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 数值特征重要性（相关系数）
    corr_cols = ['stops', 'class', 'duration', 'days_left']
    labels = {'stops': 'Stops', 'class': 'Class', 'duration': 'Duration', 'days_left': 'days_left'}
    values = [abs(df[c].corr(df['price'])) for c in corr_cols]
    lnames = [labels[c] for c in corr_cols]

    # 航空公司重要性（均值差异）
    airline_cols = [c for c in df.columns if c.startswith('airline_')]
    for col in airline_cols:
        m = df[col] == True
        if m.sum() > 10:
            diff = abs(df.loc[m, 'price'].mean() - df['price'].mean())
            values.append(diff)
            lnames.append(col.replace('airline_', ''))

    # 航线重要性
    route_cols = [c for c in df.columns if c.startswith('route_')]
    for col in route_cols:
        m = df[col] == 1
        if m.sum() > 10:
            diff = abs(df.loc[m, 'price'].mean() - df['price'].mean())
            values.append(diff)
            lnames.append(col.replace('route_', ''))

    # 归一化
    maxv = max(values) if values else 1
    norm = [v / maxv for v in values]

    # 绘制水平条形图
    colors = plt.cm.Set3(np.linspace(0, 1, len(norm)))
    bars = ax.barh(lnames, norm, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Relative Importance')
    ax.set_title('Feature Importance')
    ax.grid(True, axis='x', alpha=0.3)

    # 在每个条上标注原始重要性值
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/02_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_training(train_losses, val_losses, save_dir):
    """
    绘制训练曲线图（RMSE vs Epoch）。

    两个子图：
      左: 线性 Y 轴（直观看 RMSE 下降幅度）
      右: 对数 Y 轴（看收敛趋势和对数尺度下的下降速度）

    Args:
        train_losses: 训练损失列表（MSE）
        val_losses:   验证损失列表（MSE）
        save_dir:     保存目录
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(train_losses) + 1)
    train_rmse = [np.sqrt(l) for l in train_losses]
    val_rmse = [np.sqrt(l) for l in val_losses]

    # 线性坐标
    ax1.plot(ep, train_rmse, 'b-o', label='Train', linewidth=2, markersize=4)
    ax1.plot(ep, val_rmse, 'r-s', label='Val', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Loss Curve (linear)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 对数坐标
    ax2.semilogy(ep, train_rmse, 'b-o', label='Train', linewidth=2, markersize=4)
    ax2.semilogy(ep, val_rmse, 'r-s', label='Val', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE (log)')
    ax2.set_title('Loss Curve (log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/03_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_days_left_analysis(model, X_test_mean, ym, ys, save_dir):
    """
    分析 days_left（距起飞天数）对预测价格的影响。

    方法：
      1. 取测试集的平均特征作为基准
      2. 固定其他特征，仅改变 days_left 特征（索引 3）
      3. 用模型预测不同 days_left 下的价格
      4. 找出价格最低的 days_left

    Args:
        model:      训练好的模型
        X_test_mean: 测试集每个特征的平均值
        ym:         价格均值（用于逆变换）
        ys:         价格标准差（用于逆变换）
        save_dir:   保存目录

    Returns:
        days_range:  测试的 days_left 值数组
        preds:       对应的预测价格（原始尺度）
        min_idx:     最低价对应的索引
    """
    # 在归一化空间中的 days_left 取值范围
    days_range = np.linspace(-1.8, 1.7, 50)

    # 基准特征（所有其他特征取均值）
    base = X_test_mean.reshape(1, 1, -1)
    preds = []

    for dl in days_range:
        feat = base.copy()
        feat[0, 0, 3] = dl  # 索引 3 对应 days_left
        p = model.forward(feat)
        # 逆变换回原始价格尺度
        preds.append(p[0, 0, 0] * ys + ym)

    preds = np.array(preds)

    # 绘制曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days_range, preds, 'b-', linewidth=3, label='Prediction')
    ax.fill_between(days_range, preds * 0.85, preds * 1.15, alpha=0.15,
                    color='blue', label='Range (+/-15%)')
    ax.set_xlabel('days_left')
    ax.set_ylabel('Predicted Price')
    ax.set_title('days_left vs Predicted Price')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 标注最低价格点
    min_idx = np.argmin(preds)
    ax.annotate(f'Min: {preds[min_idx]:.0f}',
                xy=(days_range[min_idx], preds[min_idx]),
                xytext=(days_range[min_idx] + 0.3, preds[min_idx] + 300),
                fontsize=11, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.savefig(f'{save_dir}/04_days_left_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    return days_range, preds, min_idx


def plot_booking_timeline(model, X_test_mean, ym, ys, save_dir):
    """
    绘制预订时间线：从起飞前 90 天到起飞当天，价格的变化趋势。

    目的：找出最优购票时机（最低价对应的天数）。

    Args:
        model:      训练好的模型
        X_test_mean: 测试集特征均值
        ym:         价格均值（逆变换）
        ys:         价格标准差（逆变换）
        save_dir:   保存目录

    Returns:
        opt_day:       最优提前购票天数
        opt_price:     最优价格
        price_timeline: 完整的价格变化序列
    """
    # 实际天数：从起飞前 90 天到当天
    actual_days = np.linspace(90, 0, 60)

    # 将实际天数映射到归一化空间
    # 假设 days_left 范围: 90 天 → 0 天
    dl_min, dl_max = -1.82, 1.69
    norm_days = dl_min + (actual_days / 90) * (dl_max - dl_min)

    # 基准特征
    base = X_test_mean.reshape(1, 1, -1)
    price_timeline = []

    for dl in norm_days:
        feat = base.copy()
        feat[0, 0, 3] = dl
        p = model.forward(feat)
        price_timeline.append(p[0, 0, 0] * ys + ym)

    price_timeline = np.array(price_timeline)

    # 绘制
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual_days, price_timeline, 'b-', linewidth=3,
            marker='o', markersize=4, markevery=5)

    # 标注最优购票点
    min_d = np.argmin(price_timeline)
    ax.plot(actual_days[min_d], price_timeline[min_d], 'r*', markersize=20,
            label=f'Optimal: {actual_days[min_d]:.0f} days before')
    ax.axvspan(actual_days[min_d] - 5, actual_days[min_d] + 5, alpha=0.2,
               color='green', label='Optimal window (+/-5 days)')

    ax.set_xlabel('Days before departure')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Price Trend by Booking Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # X 轴反转：从左到右是从 90 天到 0 天

    plt.tight_layout()
    plt.savefig(f'{save_dir}/05_booking_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()

    return actual_days[min_d], price_timeline[min_d], price_timeline


def plot_feature_sensitivity(model, X_test_mean, ym, ys, save_dir):
    """
    特征敏感性分析：固定其他特征，分别改变每个特征，观察价格变化。

    4 个子图，每个对应一个核心特征：
      Stops, Class, Duration, days_left

    通过观察每个特征的预测价格曲线，可以理解模型学到的特征-价格关系。

    Args:
        model:       训练好的模型
        X_test_mean: 测试集特征均值
        ym:          价格均值
        ys:          价格标准差
        save_dir:    保存目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feature Sensitivity Analysis', fontsize=14, fontweight='bold')

    # 4 个核心特征的名称、索引、取值范围
    feat_names = ['Stops', 'Class', 'Duration', 'days_left']
    feat_idx = [0, 1, 2, 3]
    feat_ranges = [
        np.arange(0, 4),                     # Stops: 0~3
        np.array([0, 1]),                     # Class: 0 或 1
        np.linspace(0, 1, 30),               # Duration: 归一化 0~1
        np.linspace(-1.8, 1.7, 50),          # days_left: 归一化范围
    ]

    for ax, name, idx, fr in zip(axes.flatten(), feat_names, feat_idx, feat_ranges):
        base = X_test_mean.reshape(1, 1, -1)
        preds = []

        for v in fr:
            feat = base.copy()
            feat[0, 0, idx] = v
            p = model.forward(feat)
            preds.append(p[0, 0, 0] * ys + ym)

        preds = np.array(preds)
        ax.plot(fr, preds, linewidth=2, color='#2196F3')
        ax.fill_between(fr, preds * 0.9, preds * 1.1, alpha=0.1, color='#2196F3')
        ax.set_xlabel(f'{name}')
        ax.set_ylabel('Predicted Price')
        ax.set_title(f'{name} Effect')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/06_feature_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_prediction_vs_actual(model, X_test, y_test, ym, ys, save_dir):
    """
    绘制预测值 vs 实际值的散点图。

    同时计算并标注以下指标：
      - RMSE: 均方根误差
      - MAE:  平均绝对误差
      - R²:   决定系数（1 = 完美拟合，0 = 和均值持平，<0 = 比均值还差）

    理想情况下，散点应紧密分布在 y=x（红色虚线）附近。

    Args:
        model:       训练好的模型
        X_test:      测试特征
        y_test:      测试标签（标准化后）
        ym:          价格均值
        ys:          价格标准差
        save_dir:    保存目录

    Returns:
        rmse, r2, mae: 三个评估指标
    """
    # 批量预测
    xb = X_test.reshape(-1, 1, INPUT_DIM)
    pred = model.forward(xb)

    # 逆变换回原始价格尺度
    pred_flat = pred.flatten() * ys + ym
    actual_flat = y_test * ys + ym

    # 计算评估指标
    rmse = np.sqrt(np.mean((actual_flat - pred_flat) ** 2))
    # R² = 1 - SS_res / SS_tot
    r2 = 1 - np.sum((actual_flat - pred_flat) ** 2) / np.sum((actual_flat - actual_flat.mean()) ** 2)
    mae = np.mean(np.abs(actual_flat - pred_flat))

    # 绘制散点图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual_flat, pred_flat, alpha=0.3, s=10, color='#2196F3')

    # y=x 对角线
    lims = [min(actual_flat.min(), pred_flat.min()), max(actual_flat.max(), pred_flat.max())]
    ax.plot(lims, lims, 'r--', linewidth=2)

    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Predicted vs Actual')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')  # 等比例

    # 在图表上标注指标
    ax.text(0.05, 0.95,
            f'RMSE: {rmse:.0f}\nMAE:  {mae:.0f}\nR²:   {r2:.3f}',
            transform=ax.transAxes, fontsize=12, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{save_dir}/07_prediction_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    return rmse, r2, mae


# =============================================================================
# 5. Main — 主函数
# =============================================================================

def train_cabin(cabin, seed=42):
    """
    训练指定舱位的独立模型并生成所有可视化图表。

    完整流程（8 个步骤）：
      [1/8] 加载数据：读取并合并所有航线数据，进行标准化
      [2/8] 数据可视化：数据分布 + 特征重要性
      [3/8] 训练模型：Transformer Encoder 回归
      [4/8] 训练曲线：RMSE vs Epoch
      [5/8] days_left 分析：价格对提前购票天数的敏感性
      [6/8] 预订时间线：最优购票时机
      [7/8] 特征敏感性：各特征对价格的影响
      [8/8] 预测评估：预测 vs 实际散点图

    Args:
        cabin: 舱位类型（'economy' 或 'business'）
        seed:  随机种子

    Returns:
        results: 包含评估指标和训练数据的字典
    """
    np.random.seed(seed)
    cabin_dir = f'results_{cabin}'
    os.makedirs(cabin_dir, exist_ok=True)

    cabin_label = 'Economy' if cabin == 'economy' else 'Business'
    print("=" * 60)
    print(f" Flight Price Prediction - Transformer (Pure NumPy)")
    print(f" Cabin: {cabin_label} (independent training)")
    print("=" * 60)

    # [1/8] 加载数据
    print(f"\n[1/8] Loading all routes ({cabin})...")
    df_train, df_test = load_all_routes(cabin=cabin)
    X_train, y_train, X_test, y_test, ym, ys = make_tensors(df_train, df_test)
    print(f"  Train: {len(df_train)}, Test: {len(df_test)}")
    print(f"  Features: {INPUT_DIM}")
    print(f"  Price range: [{y_train.min()*ys+ym:.0f}, {y_train.max()*ys+ym:.0f}]")

    # [2/8] 数据可视化
    print(f"\n[2/8] Data visualization...")
    plot_data(df_train, f'All Routes Combined ({cabin_label})', cabin_dir)
    plot_feature_importance(df_train, cabin_dir)

    # [3/8] 训练模型
    print(f"\n[3/8] Training Transformer ({cabin_label})...")
    model = TransformerEncoder(
        INPUT_DIM,
        d_model=64,      # 模型维度
        n_layers=3,      # 3 层编码器
        n_heads=4,       # 4 个注意力头
        hidden_dim=128,  # FFN 隐藏层维度
        dropout=0.1      # Dropout 10%
    )
    train_losses, val_losses = train(
        model, X_train, y_train, X_test, y_test,
        epochs=50,       # 训练 50 轮
        lr=0.02,         # 学习率
        batch_size=64    # 每批 64 个样本
    )

    # [4/8] 训练曲线
    print(f"\n[4/8] Training curves...")
    plot_training(train_losses, val_losses, cabin_dir)

    # [5/8] days_left 分析
    print(f"\n[5/8] days_left analysis...")
    X_test_mean = X_test.mean(0)
    days_range, preds, min_idx = plot_days_left_analysis(model, X_test_mean, ym, ys, cabin_dir)

    # [6/8] 预订时间线
    print(f"\n[6/8] Booking timeline...")
    opt_day, opt_price, timeline = plot_booking_timeline(model, X_test_mean, ym, ys, cabin_dir)

    # [7/8] 特征敏感性
    print(f"\n[7/8] Feature sensitivity...")
    plot_feature_sensitivity(model, X_test_mean, ym, ys, cabin_dir)

    # [8/8] 预测评估
    print(f"\n[8/8] Prediction evaluation...")
    rmse, r2, mae = plot_prediction_vs_actual(model, X_test, y_test, ym, ys, cabin_dir)

    # 按航线统计
    print(f"\n {cabin_label} Per-route statistics:")
    for route in ROUTES:
        mask = df_test['route_' + route] == 1
        if mask.sum() > 0:
            route_test = df_test[mask]
            print(f"    {route}: count={len(route_test)}, mean_price={route_test['price'].mean():.0f}")

    # 计算最优购票的节省比例
    savings = timeline[0] - opt_price  # 90 天前 vs 最优时机的价格差
    savings_pct = savings / timeline[0] * 100

    # 打印总结
    print(f"\n{'=' * 60}")
    print(f" {cabin_label} Training Complete! Key Findings:")
    print(f"{'=' * 60}")
    print(f"  RMSE: {rmse:.0f}")
    print(f"  R²:   {r2:.3f}")
    print(f"  MAE:  {mae:.0f}")
    print(f"  Min predicted price: {preds.min():.0f} (days_left={days_range[min_idx]:.2f})")
    print(f"  Max predicted price: {preds.max():.0f} (days_left={days_range[np.argmax(preds)]:.2f})")
    print(f"  Optimal booking: {opt_day:.0f} days before departure")
    print(f"  90 days vs optimal: savings {savings:.0f} ({savings_pct:.1f}%)")
    print(f"\n  Charts saved to {cabin_dir}/:")
    for f in sorted(os.listdir(cabin_dir)):
        if f.endswith('.png'):
            print(f"    - {cabin_dir}/{f}")
    print(f"{'=' * 60}")

    return {
        'cabin': cabin,
        'rmse': rmse, 'r2': r2, 'mae': mae,
        'opt_day': opt_day, 'opt_price': opt_price,
        'train_losses': train_losses, 'val_losses': val_losses,
    }


def main():
    """
    主函数：依次训练经济舱和商务舱模型，并对比结果。

    流程：
      1. 经济舱独立训练
      2. 商务舱独立训练
      3. 对比两个舱位的 RMSE、R²、MAE、最优购票时机等
    """
    # 经济舱独立训练
    econ_results = train_cabin('economy', seed=42)

    print("\n\n")

    # 商务舱独立训练
    busi_results = train_cabin('business', seed=42)

    # 对比总结
    print("\n" + "=" * 60)
    print(" 经济舱 vs 商务舱 - 对比总结")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Economy':>12} {'Business':>12}")
    print(f"  {'-'*51}")
    print(f"  {'RMSE':<25} {econ_results['rmse']:>12.0f} {busi_results['rmse']:>12.0f}")
    print(f"  {'R²':<25} {econ_results['r2']:>12.3f} {busi_results['r2']:>12.3f}")
    print(f"  {'MAE':<25} {econ_results['mae']:>12.0f} {busi_results['mae']:>12.0f}")
    print(f"  {'Optimal Booking (days)':<25} {econ_results['opt_day']:>12.0f} {busi_results['opt_day']:>12.0f}")
    print(f"  {'Optimal Price':<25} {econ_results['opt_price']:>12.0f} {busi_results['opt_price']:>12.0f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
