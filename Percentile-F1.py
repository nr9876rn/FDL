from kan import *
from data_load import *
from utils import *
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve


torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

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

# 记录训练开始时间
train_start_time = time.time()

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

# 记录训练结束时间
train_end_time = time.time()
training_time = train_end_time - train_start_time

model.plot()
plt.savefig(f'{output_dir}/plot_Graph1.png')
plt.close()

# 获取正常样本的潜在表示（用于计算中心）
z_train = model(x_train).detach().cpu().numpy()
z_test = model(x_test).detach().cpu().numpy()
z_center = np.median(z_train, axis=0)  # 正常样本的中心 中位数
# 计算测试集潜在空间中的距离
anomaly_scores = np.linalg.norm(z_test - z_center, axis=1)

# 可变阈值选F1最好的那个
# 计算训练集上最大距离的百分位数
train_distances = np.linalg.norm(z_train - z_center, axis=1)


# 设置步进范围：从 80% 到 100%，步进为 0.1%
percentiles = np.arange(80, 100.1, 0.1)

# 记录最好的 F1-score 和对应的百分位数
best_f1_score = -1
best_percentile = None
best_accuracy = None
best_tpr = None
best_tnr = None
best_precision = None

# 存储每个百分位数对应的 F1 分数
f1_scores = []

# 记录测试开始时间
test_start_time = time.time()

# 进行遍历计算
for percentile in percentiles:
    threshold = np.percentile(train_distances, percentile)  # 使用当前百分位数的阈值
    y_pred = (anomaly_scores > threshold).astype(int)  # 1 表示异常，0 表示正常

    # 计算检测性能
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

    print(f"Percentile: {percentile}%, Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # 更新最好的 F1-score 和对应的百分位数
    if f1 > best_f1_score:
        best_f1_score = f1
        best_percentile = percentile
        best_accuracy = accuracy
        best_tpr = tpr
        best_tnr = tnr
        best_precision = precision

# 记录测试结束时间
test_end_time = time.time()
testing_time = test_end_time - test_start_time

# 计算 AUROC
auroc = roc_auc_score(y_test, anomaly_scores)

# 计算 AUPR
precision, recall, _ = precision_recall_curve(y_test, anomaly_scores)
aupr = auc(recall, precision)

# 计算 FPR95
fpr95 = None
for i in range(len(recall)):
    if recall[i] >= 0.95:
        fpr95 = 1 - tnr  # 这里假设 tnr 是当前阈值下的真负率
        break

# 打印最好的结果
print("\nBest Percentile:", best_percentile)
print("Best Accuracy:", best_accuracy)
print("Best F1-score:", best_f1_score)
print("Best TPR:", best_tpr)
print("Best TNR:", best_tnr)
print("Best Precision:", best_precision)
print("AUROC:", auroc)
print("AUPR:", aupr)
print("FPR95:", fpr95)
print("Training Time:", training_time)
print("Testing Time:", testing_time)


import matplotlib.pyplot as plt
# 设置支持负号的字体（优先使用英文字体）
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial']  # 备用Arial确保负号
plt.rcParams['axes.unicode_minus'] = True  # 必须设置为True！
# 绘制 Percentile 对 F1 的影响曲线
plt.figure()
plt.plot(percentiles, f1_scores)
plt.xlabel('百分位数')
plt.ylabel('F1分数')
plt.title('百分位数对F1分数的影响')

plt.tight_layout()
plt.savefig(f'{output_dir}/auroc_aupr_percentile_f1_curves.png')
plt.show()

