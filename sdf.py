import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========== 1. 数据 ==========
X = np.array([
    [2, 2],     # 正类
    [3, 3],
    [1, -1],    # 负类
    [2, -2]
])
y = np.array([1, 1, -1, -1])

# ========== 2. 初始化 ==========
theta = np.zeros(2)
b = 0

history = []  # 存储每一步的参数

# ========== 3. 感知机训练 ==========
for epoch in range(10):
    for i in range(len(X)):
        xi = X[i]
        yi = y[i]

        if yi * (np.dot(theta, xi) + b) <= 0:
            theta = theta + yi * xi
            b = b + yi
            history.append((theta.copy(), b))

# ========== 4. 画图 ==========
fig, ax = plt.subplots()

# 画数据点
for i in range(len(X)):
    if y[i] == 1:
        ax.scatter(X[i,0], X[i,1])
    else:
        ax.scatter(X[i,0], X[i,1])

x_vals = np.linspace(-3, 4, 100)
line, = ax.plot([], [])

ax.set_xlim(-3, 4)
ax.set_ylim(-3, 4)
ax.set_title("Perceptron Decision Boundary")

# ========== 5. 动画函数 ==========
def update(frame):
    theta, b = history[frame]

    if theta[1] != 0:
        y_vals = -(theta[0] * x_vals + b) / theta[1]
        line.set_data(x_vals, y_vals)

    ax.set_title(f"Step {frame+1} | theta={theta}, b={b}")
    return line,

# ========== 6. 动画 ==========
ani = FuncAnimation(fig, update, frames=len(history), interval=800, repeat=False)

plt.show()