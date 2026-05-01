import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    """线性回归 - 梯度下降实现"""
    def __init__(self,learning_rate=0.1,n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations#self.n_iterations	传入的 n_iterations	存储最大迭代次数
        self.theta = None#线的参数（斜率和截距），刚开始不知道，所以是空
        self.losses = []

    def fit(self,X,y):
        """
        X: (m,n) m个样本, n个特征
        y: (m,) m个标签
        """
        m,n = X.shape#m, n 两个变量，利用元组解包分别接收 X.shape 返回的两个数字。
        # 初始化参数（全0）
        self.theta = np.zeros(n)

        for i in range(self.n_iterations):#range(self.n_iterations)	Python 内置函数，生成一个从 0 到 self.n_iterations-1 的整数序列。
            # 1. 预测（前向传播）
            predictions = X @ self.theta# (m,)
            # 2. 误差
            errors = predictions - y    # (m,)
            # 3. 梯度（你上午推导的公式！）
            gradient = X.T @ errors / m # (n,)
            # 4. 更新参数
            self.theta -= self.lr*gradient#往梯度的反方向走（因为我们要最小化损失，不是最大化）
            # 5. 记录损失（MSE/2）
            loss = np.mean(errors**2) / 2#np.mean(...)：取平均
            self.losses.append(loss)#self.losses.append(loss)：把这轮的错误记到"错题本"里
        return self 
    
    def predict(self,X):
        """预测"""
        return X @ self.theta
    
    def score(self,X,y):
        """计算R^2分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y-y_pred)**2)#残差平方和 (SS_res)，（预测误差）
        ss_tot = np.sum((y-np.mean(y))**2)#总平方和 (SS_tot)，（数据本身的波动）
        return 1 - ss_res / ss_tot#决定系数 R^2 

# ========== 生成假数据：y = 2x + 1 + 噪声 ==========
np.random.seed(42)# 固定随机种子，结果可复现
# 100个样本，1个特征
X_raw = 2*np.random.rand(100,1)#np.random.rand	函数名：从 均匀分布 [0, 1) 中生成随机数。(100, 1)	参数，指定生成数组的形状。这里要求生成一个 100 行、1 列 的二维数组。返回值（未乘以 2 前）	一个形状为 (100, 1) 的 NumPy 数组，每个元素独立地取自 [0, 1) 上的均匀分布。
# 真实关系：y = 2*x + 1
# 加上高斯噪声（模拟真实数据）
noise = np.random.randn(100,1)*0.3# 标准差0.3的噪声
y = 2 * X_raw + 1 + noise  
# 为了有截距项，给X加一列1
X = np.c_[np.ones((100,1)),X_raw]# (100, 2)，第一列全是1
# 把y压平成一维
y = y.ravel()  # (100,)

print("X形状:",X.shape) # (100, 2)
print("y形状:",y.shape) # (100,)

# ========== 训练 ==========
model = LinearRegressionGD(learning_rate=0.1,n_iterations=500)
model .fit(X,y)

print("学习到的参数:", model.theta)
print("真实参数应该是: [1, 2](截距1,斜率2)")
print("R²分数:", model.score(X, y))

# ========== 可视化 ==========
plt.figure(figsize=(10,4))

# 子图1：数据点和拟合直线
plt.subplot(1,2,1)#plt.subplot(1,2,1) 将画布分为 1 行 2 列，并激活左侧第一个子图，后续绘图命令将在该子图中执行。它是 Matplotlib 中创建多子图的经典过程式方法。
plt.scatter(X_raw,y,alpha=0.6,label='data')#plt.scatter 是 Matplotlib 的散点图函数.X_raw横坐标数据,y纵坐标数据。

# 画拟合直线
x_line = np.linspace(0,2,100)#函数名，linearly space（线性空间），用于生成在指定区间内均匀分布的数值序列。第二个参数：结束值（stop），序列的终点。注意：linspace 包含终点（与 range 不同）。
y_line = model.theta[0] + model.theta[1] * x_line#model.theta[0]参数数组的第一个元素，截距;model.theta[1]参数数组的第二个元素，斜率（slope）。
#plot	函数名，用于绘制折线图（或点线图）。它根据给定的 x 和 y 坐标，将点按顺序连接起来
plt.plot(x_line, y_line, 'r-', linewidth=2, label='Fitted line')#'r-'一个格式字符串，同时指定线条的颜色和线型。'r'--red；'-'--实线。
plt.xlabel('x')#设置x轴标签
plt.ylabel('y')
plt.legend()#识别之前所有绘图命令中的 label 参数，将它们组合成一个图例框，放置在图表的合适位置（默认自动选择位置）。
plt.title('Linear Regression Fit')

# 子图2：损失曲线
plt.subplot(1,2,2)
plt.plot(model.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss J(θ)')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)#grid()函数名，用于控制图表网格线的显示与样式。True 表示显示网格线；False（或不写）表示不显示网格线。

plt.tight_layout()#.tight_layout()	方法，自动调整子图布局。
plt.savefig('day03_linear_regression.png')  # 保存图片
plt.show()#显示当前所有已绘制的图形