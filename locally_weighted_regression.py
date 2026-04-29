import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io

# ========== 1. 数据下载 ==========
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 
           'weight', 'acceleration', 'model_year', 'origin', 'car_name']

print("正在下载 Auto MPG 数据集...")
response = requests.get(url)
# 处理多空格分隔
data = pd.read_csv(io.StringIO(response.text),
                   sep='\s+',   # 使用正则表达式：一个或多个空白字符
                   header=None,
                   names=columns,
                   na_values='?')

# ========== 2. 数据清洗 ==========
print(f"原始数据量: {len(data)}")
data = data.dropna()  # 删除 horsepower 中的缺失值
print(f"清洗后数据量: {len(data)}")

# 为演示局部加权，选取单特征：weight -> mpg
X_raw = data['weight'].values.reshape(-1, 1)
y_raw = data['mpg'].values

# 标准化（便于观察）
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X_raw).flatten()
y = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

print(f"\n特征范围: [{X.min():.2f}, {X.max():.2f}]")
print(f"目标范围: [{y.min():.2f}, {y.max():.2f}]")

# 可视化原始数据
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='blue', alpha=0.5, s=30, label='Data Points')
plt.xlabel('Weight (standardized)')
plt.ylabel('MPG (standardized)')
plt.title('Auto MPG: Weight vs MPG')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('E:/AI/ufldl_practice/auto_mpg_scatter.png', dpi=150)
plt.show()
# ========== 3. UFLDL 局部加权线性回归实现 ==========

class UFLDL_LWLR:
    """
    局部加权线性回归（Locally Weighted Linear Regression）
    Non-parametric learning algorithm
    """
    def __init__(self, tau=0.5):
        """
        tau: 带宽参数（bandwidth）
             - tau 越小，权重衰减越快，拟合越"局部"（容易过拟合）
             - tau 越大，权重衰减越慢，趋近普通线性回归（容易欠拟合）
        """
        self.tau = tau
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """存储训练数据（non-parametric 特性：没有显式训练参数）"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _compute_weights(self, x_query):
        """
        计算高斯权重核
        w(i) = exp(-(x(i) - x)^2 / (2*tau^2))
        """
        diff = self.X_train - x_query
        # 对角权重矩阵的对角线元素
        weights = np.exp(-(diff ** 2) / (2 * self.tau ** 2))
        return np.diag(weights)
    
    def predict(self, x_query):
        """
        对单个查询点进行预测
        θ = (X^T W X)^(-1) X^T W y
        """
        X = np.c_[np.ones(len(self.X_train)), self.X_train]  # 加截距
        x_q = np.array([1, x_query])
        
        W = self._compute_weights(x_query)
        
        # 加权正规方程
        XTX = X.T @ W @ X
        
        # 处理奇异矩阵
        try:
            theta = np.linalg.inv(XTX) @ X.T @ W @ self.y_train
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(XTX) @ X.T @ W @ self.y_train
            
        return x_q @ theta
    
    def predict_batch(self, X_test):
        """批量预测（对每个点单独计算）"""
        return np.array([self.predict(x) for x in X_test])


# ========== 4. 对比实验：普通线性回归 vs 局部加权回归 ==========

# 普通最小二乘（用于对比）
class OrdinaryLeastSquares:
    def fit(self, X, y):
        self.X = np.c_[np.ones(len(X)), X]
        self.theta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ y
        return self
    
    def predict(self, X):
        X = np.c_[np.ones(len(X)), X]
        return X @ self.theta

# 排序用于平滑绘图
sort_idx = np.argsort(X)
X_sorted = X[sort_idx]
y_sorted = y[sort_idx]

# 普通线性回归
ols = OrdinaryLeastSquares()
ols.fit(X, y)
y_ols = ols.predict(X_sorted)

# 不同 tau 值的 LWLR
taus = [0.1, 0.5, 2.0]
colors = ['red', 'green', 'purple']
lwlr_models = {}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 子图1：普通线性回归
axes[0,0].scatter(X, y, c='blue', alpha=0.4, s=30, label='Data')
axes[0,0].plot(X_sorted, y_ols, 'k-', linewidth=2, label='OLS')
axes[0,0].set_title('Ordinary Least Squares (Global Linear Fit)')
axes[0,0].set_xlabel('Weight (standardized)')
axes[0,0].set_ylabel('MPG (standardized)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 子图2-4：不同 tau 的 LWLR
positions = [(0,1), (1,0), (1,1)]
for tau, color, pos in zip(taus, colors, positions):
    print(f"\n正在计算 LWLR (tau={tau})...")
    lwlr = UFLDL_LWLR(tau=tau)
    lwlr.fit(X, y)
    y_lwlr = lwlr.predict_batch(X_sorted)
    lwlr_models[tau] = y_lwlr
    
    axes[pos].scatter(X, y, c='blue', alpha=0.4, s=30, label='Data')
    axes[pos].plot(X_sorted, y_lwlr, color=color, linewidth=2, 
                   label=f'LWLR (τ={tau})')
    axes[pos].set_title(f'Locally Weighted Regression (τ={tau})')
    axes[pos].set_xlabel('Weight (standardized)')
    axes[pos].set_ylabel('MPG (standardized)')
    axes[pos].legend()
    axes[pos].grid(True, alpha=0.3)
    
    # 计算 MSE
    mse = np.mean((y_sorted - y_lwlr)**2)
    axes[pos].text(0.05, 0.95, f'MSE: {mse:.4f}', 
                   transform=axes[pos].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('E:/AI/ufldl_practicelwlr_comparison.png', dpi=150)
plt.show()

# ========== 5. 带宽参数 tau 的影响分析 ==========
plt.figure(figsize=(12, 7))
plt.scatter(X, y, c='lightgray', alpha=0.5, s=30, label='Data')
plt.plot(X_sorted, y_ols, 'k--', linewidth=2, label='OLS (Global)')

for tau, color in zip(taus, colors):
    plt.plot(X_sorted, lwlr_models[tau], color=color, linewidth=2.5,
             label=f'LWLR τ={tau}')

plt.xlabel('Weight (standardized)', fontsize=12)
plt.ylabel('MPG (standardized)', fontsize=12)
plt.title('Effect of Bandwidth Parameter τ on LWLR Fit', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig('E:/AI/ufldl_practice/lwlr_tau_analysis.png', dpi=150)
plt.show()

print("\n=== 实验结论 ===")
print("τ=0.1: 权重衰减极快，拟合曲线非常曲折（高方差/过拟合）")
print("τ=0.5: 适中的局部性，较好地捕捉非线性趋势")
print("τ=2.0: 权重衰减慢，接近全局线性回归（高偏差/欠拟合）")