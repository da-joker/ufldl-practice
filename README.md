# UFLDL 学习记录

从零手写机器学习算法。

## 进度

- [x] Day 1: Python基础 - 变量、列表、字典 ✅
- [x] Day 2: 循环判断、函数、类、NumPy基础 ✅
- [x] Day 3: 线性回归梯度下降实现 ✅
- [x] Day 4: 梯度下降调参+特征缩放+逻辑回归理论 ✅
- [ ] Day 5: 手写逻辑回归类（Heart Disease数据集）

## 环境

- Python 3.11
- NumPy, Pandas, Matplotlib

## 关键实验

### Day 4 发现
- 学习率 α=1.0 导致 loss 爆炸（1e68），α=0.01 最稳定
- 特征缩放前：梯度方向混乱，100轮发散
- 特征缩放后：20轮平滑收敛
- 数学本质：X^T X 的条件数决定收敛速度

## 运行方式

```bash
cd week1
python day04.py