from kan import *
from data_load import *
import seaborn as sns
from utils import *
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 确保图像保存目录存在
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""
'Graph': ['ethereum'], 34
'BNaT': ['w1']  19
"""

dataset = 'Graph'
subset = 'ethereum'
X_train, X_valid, y_train, y_valid = load_data(dataset, subset, mode='train')
X_test, y_test = load_data(dataset, subset, mode='test')

# 创建对象，按列归一化数据
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



# 固定阈值
# 获取训练集最大距离的 99% 百分位数作为阈值
threshold = np.percentile(np.linalg.norm(z_train - z_center, axis=1), 99.9)
# 预测异常
y_pred = (anomaly_scores > threshold).astype(int)  # 1 表示异常，0 表示正常

# 计算检测性能
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))


# 可视化
# 颜色映射: 正常(0) -> 绿色, 异常(1) -> 红色
import matplotlib.pyplot as plt
threshold = 0.021
import matplotlib as mpl
# 将负号替换为ASCII的减号（-）
mpl.rcParams['axes.unicode_minus'] = False  # 禁用Unicode负号
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun'] # 使用宋体

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

