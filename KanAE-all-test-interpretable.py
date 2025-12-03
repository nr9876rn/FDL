# 卿雨竹
# 开发时间：2025-03-14 15:37
# 输出可解释公式和图

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


dataset = 'Graph'
subset = 'ethereum'
X_train, X_valid, y_train, y_valid = load_data(dataset, subset, mode='train')
X_test, y_test = load_data(dataset, subset, mode='test')

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
plt.show()


# 获取正常样本的潜在表示（用于计算中心）
z_train = model(x_train).detach().cpu().numpy()
z_test = model(x_test).detach().cpu().numpy()

# 符号化
lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'tan', 'abs']
model.auto_symbolic(lib=lib)
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

model.plot()
plt.show()


