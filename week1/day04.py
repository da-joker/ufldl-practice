import numpy as np
import matplotlib.pyplot as plt

# 复用昨天的 LinearRegressionGD 类
class LinearRegressionGD:
    def __init__(self,learning_rate=0.1,n_iterations=500):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.theta = None
        self.losses = []

    def fit(self,X,y):
        m,n = X.shape
        self.theta = np.zeros(n)
        for i in range(self.n_iter):
            predictions = X @ self.theta
            errors = predictions - y
            gradient = X.T @ errors / m
            self.theta -= self.lr*gradient
            loss = np.mean(errors**2) / 2
            self.losses.append(loss)
        return self

# 生成假数据（范围放大，测试学习率影响）
np.random.seed(42)   
X_raw = 10 * np.random.rand(100,1)
noise = np.random.randn(100,1)*2
y = 3 * X_raw + 5 + noise
y = y.ravel()
X = np.c_[np.ones((100,1)),X_raw]

# 测试 4 个学习率
alphas = [0.001,0.01,0.1,1.0]
colors = ['blue','green','red','purple']

plt.figure(figsize=(10,6))

for alpha,color in zip(alphas,colors):
    model = LinearRegressionGD(learning_rate=alpha, n_iterations=300)
    model.fit(X,y)
    plt.plot(model.losses,color= color,label=f'α={alpha}')
    
plt.xlabel('Iteration')
plt.ylabel('Loss J(θ)')
plt.title('Effect of Learning Rate α')
plt.legend()#plt.legend() 用于为图表添加图例，帮助识别不同数据系列，通常需要配合绘图函数中的 label 参数使用。
plt.grid(True,alpha=0.3)
plt.ylim(0, 50)   # 只看 0-50 的范围，把爆炸的紫线截断
plt.savefig('day04_learning_rates.png')
plt.show()

print("观察：")
print("- α=0.001（蓝）：下降极慢，300轮还没收敛")
print("- α=0.01（绿）：下降较慢，但稳定")
print("- α=0.1（红）：快速收敛，几十轮搞定")
print("- α=1.0（紫）：震荡甚至发散！")

# ========== 特征缩放 ==========
X_mean = np.mean(X_raw)#计算原始特征 X_raw 的均值（mean）。
X_std = np.std(X_raw)#计算原始特征的标准差（standard deviation）。
X_scaled = (X_raw - X_mean) / X_std#将每个原始值减去均值，再除以标准差，使新数据均值为 0，标准差为 1。

# 重新组装（加偏置项）
X_norm = np.c_[np.ones((100, 1)), X_scaled]
print("原始特征范围:", X_raw.min(), "~", X_raw.max())
print("标准化后范围:", X_scaled.min(), "~", X_scaled.max())

# 对比：相同学习率 α=0.1，缩放前 vs 缩放后
model_raw = LinearRegressionGD(learning_rate=0.1, n_iterations=100)
model_raw.fit(X, y)

model_norm = LinearRegressionGD(learning_rate=0.1, n_iterations=100)
model_norm.fit(X_norm, y)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(model_raw.losses)
plt.title('Before Scaling (α=0.1)')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(model_norm.losses)
plt.title('After Scaling (α=0.1)')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.tight_layout()#plt.tight_layout() 是 Matplotlib 中用来自动调整子图周围间距的函数，避免不同子图的标签、刻度、标题之间相互重叠。
plt.savefig('day04_scaling.png')
plt.show()


#逻辑回归

z = np.linspace(-8, 8, 400)
sigma = 1 / (1 + np.exp(-z))
grad = sigma * (1 - sigma)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(z, sigma); axes[0].set_title('Sigmoid σ(z)')
axes[0].scatter([0], [0.5], color='red'); axes[0].grid(True)
axes[1].plot(z, grad); axes[1].set_title("Sigmoid Derivative σ'(z)")
axes[1].scatter([0], [0.25], color='red'); axes[1].grid(True)
plt.tight_layout(); plt.show()