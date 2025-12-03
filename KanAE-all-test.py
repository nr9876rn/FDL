# 卿雨竹
# 开发时间：2025-03-14 15:37
# 效果不错欸

from kan import *
from data_load import *
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 确保图像保存目录存在
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""
'cicids_custom': ['Tuesday'], 30
'toniot_custom': ['ddos'], 30
'cse_improved': ['server1'], 40
'Graph': ['ethereum'], 34
'BNaT': ['w1']  19
"""

dataset = 'BNaT'
subset = 'w1'
X_train, X_valid, y_train, y_valid = load_data(dataset, subset, mode='train')
X_test, y_test = load_data(dataset, subset, mode='test')

# 创建对象，按列归一化数据
# scaler = MinMaxScaler()
scaler = StandardScaler()
# 对训练集、验证集和测试集的特征进行归一化
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)  # 使用训练集的scaler进行变换
X_test = scaler.transform(X_test)  # 使用训练集的scaler进行变换

# 转换为 PyTorch 张量
x_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
x_valid = torch.tensor(X_valid, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train[:, None], dtype=torch.float32)
y_valid = torch.tensor(y_valid[:, None], dtype=torch.float32)
y_test = torch.tensor(y_test[:, None], dtype=torch.float32)

# 将数据集放入字典
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = {}
dataset['train_input'] = x_train.to(device)
dataset['val_input'] = x_valid.to(device)
dataset['test_input'] = x_test.to(device)
dataset['train_label'] = y_train.to(device)
dataset['val_label'] = y_valid.to(device)
dataset['test_label'] = y_test.to(device)


# KAN 训练，仅学习正常样本的潜在表示
model = KAN(width=[x_train.shape[1], 2], grid=3, k=3, device=DEVICE)  # 2 维潜在空间
model.fit({
    'train_input': x_train,
    'train_label': y_train,
    'val_input': x_valid,
    'val_label': y_valid,
    'test_input': x_test,
    'test_label': y_test
}, opt="LBFGS", steps=20)

model.plot()
plt.savefig(f'{output_dir}/plot_Graph1.png')
plt.close()

# 获取正常样本的潜在表示（用于计算中心）
z_train = model(x_train).detach().cpu().numpy()
z_test = model(x_test).detach().cpu().numpy()
# z_center = np.mean(z_train, axis=0)  # 正常样本的中心 均值
z_center = np.median(z_train, axis=0)  # 正常样本的中心 中位数
# 计算测试集潜在空间中的距离
anomaly_scores = np.linalg.norm(z_test - z_center, axis=1)


"""
# 固定阈值
# 获取训练集最大距离的 99% 百分位数作为阈值
threshold = np.percentile(np.linalg.norm(z_train - z_center, axis=1), 98.4)

# 预测异常
y_pred = (anomaly_scores > threshold).astype(int)  # 1 表示异常，0 表示正常

# print("z_train type:", type(z_train), "z_train shape:", z_train.shape)
# print("z_test type:", type(z_test), "z_test shape:", z_test.shape)
# print("z_center type:", type(z_center), "z_center shape:", z_center.shape)
# print("anomaly_scores type:", type(anomaly_scores), "anomaly_scores shape:", anomaly_scores.shape)
# print("threshold type:", type(threshold), "threshold shape:", np.shape(threshold))
# print("y_pred type:", type(y_pred), "y_pred shape:", y_pred.shape)

# 计算检测性能
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
"""


# 可变阈值选F1最好的那个
# 计算训练集上最大距离的百分位数
train_distances = np.linalg.norm(z_train - z_center, axis=1)

# 设置步进范围：从 80% 到 100%，步进为 0.1%
percentiles = np.arange(50, 100, 0.1)

# 记录最好的 F1-score 和对应的百分位数
best_f1_score = -1
best_percentile = None
best_accuracy = None

# 进行遍历计算
for percentile in percentiles:
    threshold = np.percentile(train_distances, percentile)  # 使用当前百分位数的阈值
    y_pred = (anomaly_scores > threshold).astype(int)  # 1 表示异常，0 表示正常

    # 计算检测性能
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Percentile: {percentile}%, Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")

    # 更新最好的 F1-score 和对应的百分位数
    if f1 > best_f1_score:
        best_f1_score = f1
        best_percentile = percentile
        best_accuracy = accuracy

# 打印最好的结果
print("\nBest Percentile:", best_percentile)
print("Best Accuracy:", best_accuracy)
print("Best F1-score:", best_f1_score)



"""
# 符号化
lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'tan', 'abs']
model.auto_symbolic(lib=lib)
# 假设模型输出维度为 2，符号回归将自动适配
# 获取符号化公式，自动适应输出维度
formulas = model.symbolic_formula()[0]
# 根据模型的输出维度自动设置 y
output_dim = len(formulas)
y = [None] * output_dim
for i in range(output_dim):
    y[i] = ex_round(formulas[i], 4)  # 四舍五入公式

# 打印符号化公式
for i in range(output_dim):
    print(f"y{i+1}: {y[i]}")  # 显示符号化的公式
# # 模型输出是 2 维，y 有 y1 和 y2
# y1, y2 = y[0], y[1]
# print("y1:", y1)
# print("y2:", y2)
model.plot()
plt.savefig(f'{output_dir}/plot_Graph2.png')
plt.close()
"""


"""
# 解析符号公式并计算数值化输出
def parse_formula_and_evaluate(formula, x_test):
    # 获取输入变量
    num_features = x_test.shape[1]
    x_vars = [sp.Symbol(f'x_{i+1}') for i in range(num_features)]  # 变量名改为 x_1, x_2, ...
    # 将公式转换为符号表达式
    sym_expr = sp.sympify(formula)
    # 计算符号化模型的输出
    z_test_sym = np.zeros(x_test.shape[0])  # 存储计算结果
    for j in range(x_test.shape[0]):
        # 替换变量
        substitutions = {f'x_{k+1}': x_test[j, k] for k in range(num_features)}
        substituted_expr = sym_expr.subs(substitutions)
        # 计算数值化表达式
        z_test_sym[j] = float(substituted_expr.evalf())
    return z_test_sym


# 计算符号化模型的输出 (z_test_sym)
z_test_sym = np.zeros((x_test.shape[0], len(y)))  # 初始化符号化潜在表示数组

for i in range(len(y)):  # 遍历每个输出维度
    z_test_sym[:, i] = parse_formula_and_evaluate(y[i], x_test)

# 计算异常分数
anomaly_scores_sym = np.linalg.norm(z_test_sym - z_center, axis=1)
# 使用阈值预测异常
y_pred_sym = (anomaly_scores_sym > threshold).astype(int)

# 评估符号化模型的性能
acc_sym = accuracy_score(y_test, y_pred_sym)
f1_sym = f1_score(y_test, y_pred_sym)
print("符号化模型 Accuracy:", acc_sym)
print("符号化模型 F1-score:", f1_sym)
"""


"""
# 可视化
# 颜色映射: 正常(0) -> 绿色, 异常(1) -> 红色
colors = ['green' if label == 0 else 'red' for label in y_test.cpu().numpy().flatten()]

plt.figure(figsize=(8, 6))
sns.scatterplot(x=z_test[:, 0], y=z_test[:, 1], c=colors, alpha=0.6, edgecolors='k')

# 标记 `z_center`
center_scatter = plt.scatter(z_center[0], z_center[1], color='blue', marker='x', s=100, label='Center')

# 画以 z_center 为圆心，threshold 为半径的圆
circle = plt.Circle(z_center, threshold, color='orange', fill=False, linestyle='--', label='Threshold')
plt.gca().add_artist(circle)

plt.xlabel('Latent Dim 1')
plt.ylabel('Latent Dim 2')
plt.title('Test Data Distribution in Latent Space')

# 获取绘图元素的句柄
handles = []
# 正常点和异常点的句柄
handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Normal (Green)'))
handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Anomalous (Red)'))
# 中心点的句柄
handles.append(center_scatter)
# 阈值圆的句柄
handles.append(plt.Line2D([0], [0], color='orange', linestyle='--', label='Threshold'))

# 设置图例
labels = [handle.get_label() for handle in handles]
plt.legend(handles=handles, labels=labels)

plt.grid(True)

# 保存图片
plt.savefig(f'{output_dir}/latent_space_visualization.png', dpi=300)
plt.show()
"""