import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #核心能力：结构化数据处理，提供 DataFrame 和 Series，支持读取文件、清洗、筛选、分组等操作。
import requests #核心能力：发送 HTTP 请求，获取网络资源。
import io #核心能力：提供在内存中操作文本/二进制流的工具。

class LogisticRegression:
    """逻辑回归 - 梯度下降实现(带L2正则化)"""
    def __init__(self,lr=0.1,n_iter=1000,lambda_reg=0.1):
        self.lr = lr
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg
        self.theta = None
        self.losses = []

    def _sigmoid(self,z):
        """Sigmoid函数,带数值保护"""
        # np.clip防止exp溢出
        z_safe = np.clip(z,-250,250)
        return 1 / (1 + np.exp(-z_safe))
    
    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)

        for i in range(self.n_iter):
            # 1. 线性组合
            z = X @ self.theta
            # 2. Sigmoid激活
            h = self._sigmoid(z)
            # 3. 梯度（形式和线性回归一样！）
            gradient = X.T @ (h - y) / m
            # 4. L2正则化（不对theta_0正则化）
            gradient[1:] += (self.lambda_reg / m) * self.theta[1:] 
            # 5. 更新参数
            self.theta -= self.lr * gradient
            # 6. 交叉熵损失（加1e-5防止log(0)）
            loss = -np.mean(y * np.log(h + 1e-5) + (1 - y) * np.log(1 - h + 1e-5))
            # 加上正则化项
            reg_loss = (self.lambda_reg / (2 * m)) * np.sum(self.theta[1:]**2)
            self.losses.append(loss + reg_loss) 
        return self
    
    def predict_proba(self, X):
        """预测概率 P(y=1|x)"""
        return self._sigmoid(X @ self.theta)
    
    def predict(self, X):
        """预测类别(0或1)"""
        return (self.predict_proba(X) >= 0.5).astype(int)#.astype(int)	把布尔转成整数：True→1, False→0
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# 假数据：y = 1 if x1 + x2 > 0 else 0
np.random.seed(42)
X_fake = np.random.randn(200, 2)
X_fake_b = np.c_[np.ones((200, 1)), X_fake]
y_fake = (X_fake[:, 0] + X_fake[:, 1] > 0).astype(int)

model = LogisticRegression(lr=0.5, n_iter=500)
model.fit(X_fake_b, y_fake)
print("假数据准确率:", model.accuracy(X_fake_b, y_fake))
print("参数:", model.theta)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
           'restecg', 'thalach', 'exang', 'oldpeak', 
           'slope', 'ca', 'thal', 'target']#url参数，表示统一资源定位符（网址字符串）。

response = requests.get(url)
df = pd.read_csv(io.StringIO(response.text), header=None, names=columns)#将网络请求得到的CSV字符串包装成内存文件对象，然后用pandas读取为带指定列名的DataFrame，实现了无需落盘的数据直达。

# 清洗
df = df.replace('?', np.nan).dropna().astype(float)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
# 分离 X, y
X_raw = df.drop('target', axis=1).values#.drop()：DataFrame 方法，用于删除指定的行或列。'target'：要删除的列名，也就是标签列。删除后剩下全是特征列。axis=1：指定操作轴。axis=1 表示列方向（删除列）；若 axis=0 则是删除行。.values将表格数据转换为 NumPy 二维数组
y = df['target'].values

# 标准化
X_mean = np.mean(X_raw, axis=0)#axis=0：沿着第 0 轴（即行方向）求平均，结果是对每一列分别求均值，返回一个长度为 n 的向量。
X_std = np.std(X_raw, axis=0)#同理，计算每一列的标准差，得到 X_std 向量。
X_scaled = (X_raw - X_mean) / X_std#这一步就是 Z-score 标准化：对每个特征，减去均值后除以标准差，使得处理后的特征均值为 0、标准差为 1。

# 加偏置
X = np.c_[np.ones((len(X_scaled), 1)), X_scaled]

# 划分
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]#X[:split]  Python 切片语法，取前 split 行（索引 0 到 split-1）。X[split:]取从索引 split 开始到最后的所有行。
y_train, y_test = y[:split], y[split:]

print("训练集:", len(X_train), "测试集:", len(X_test))
print("类别分布:", np.bincount(y))

# ========== Heart Disease 训练 ==========
model = LogisticRegression(lr=0.5, n_iter=1000, lambda_reg=0.1)
model.fit(X_train, y_train)

train_acc = model.accuracy(X_train, y_train)
test_acc = model.accuracy(X_test, y_test)

print(f"\n训练准确率: {train_acc:.4f}")
print(f"测试准确率: {test_acc:.4f}")

# 详细指标
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test)
print(f"\n混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")
print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])}")

# ========== 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 左图：损失曲线
axes[0].plot(model.losses, color='blue', linewidth=2)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('Logistic Regression: Training Loss')
axes[0].grid(True, alpha=0.3)
# 右图：特征权重
feature_names = ['Intercept'] + list(df.drop('target', axis=1).columns)
colors_bar = ['gray'] + ['red' if t > 0 else 'blue' for t in model.theta[1:]]
axes[1].barh(feature_names, model.theta, color=colors_bar, alpha=0.7)
axes[1].set_xlabel('Weight Value')
axes[1].set_title('Learned Parameters (θ)')
axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('day05_heart_disease.png', dpi=150)
plt.show()