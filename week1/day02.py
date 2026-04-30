# 循环 for----打印1到10
for i in range(1,11):
    print(i)
# 打印1-100的奇数
for i in range(1,101,2): # 从1开始，步长为2
    print(i)
# 计算列表平均值
scores = [83,92,78,90,88]
total = 0
for s in scores:
    total += s # total = total + s
average = total / len(scores)
print("平均分:",average)
# 找最大值
max_scores = scores[0]
for s in scores:
    if s > max_scores:
        max_scores = s
print("最高分:",max_scores)
# ========== 判断 if/elif/else ==========
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "D"
print("等级:",grade)
# ========== 循环+判断结合 ==========
# 打印列表中的偶数
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for num in numbers:
    if num % 2 ==0: # %是取余数，num%2==0表示偶数
        print(num,"是偶数")
# break: 遇到第一个负数就停止
data = [3,5,8,-1,2,4]
for d in data:
    if d < 0:
        print("遇到负数，停止")
        break
    print(d)
# continue: 跳过偶数，只打印奇数
for num in range(10):
    if num % 2 ==0:
        continue# 跳过本次循环，继续下一次
    print(num)
# ========== 函数 ==========
# 基本函数
def add(a,b):
    """返回a+b"""
    return a+b

print("3+5=",add(3,5))

# 默认参数----name：必需参数，调用时必须提供。默认参数必须放在必需参数之后
def greet(name,greeting="你好"):#greeting="你好"：默认参数，如果调用时未提供第二个参数，则自动使用 "你好"。
    return greeting + "," + name

print(greet("小明")) # 使用默认值
print(greet("Tom","Hello"))# 传入新值

# 为逻辑回归铺垫：Sigmoid函数
import numpy as np
def sigmoid(z):
    """sigmoid函数,输入z可以是数字或数组"""
    return 1 / (1 + np.exp(-z))
# 测试
print("sigmoid(0)=",sigmoid(0))
print("sigmoid(2)=",sigmoid(2))
print("sigmoid(-2)=",sigmoid(-2))
# 测试数组
z_array = np.array([0,2,-2])#array()：NumPy 中创建数组的函数。
print("sigmoid(数组)=",sigmoid(z_array))
# 你的练习：写一个power函数
def power(base,exp):
    """计算base的exp次方"""
    return base**exp

# 测试
print("2^3=",power(2,3))

# ========== 类 ==========
class Student:
    """"学生类"""
    def __init__(self,name,score):
        #self 代表即将创建的实例对象自身（可以理解为“这个学生”），通过 self 可以为该实例绑定属性。
        self.name = name# 这个学生的名字。self.name = name 将参数 name 存储到实例属性 name 中。之后可以通过 对象.name 访问。
        self.score = score # 这个学生的分数
    
    def print_info(self):
        print(f"{self.name}的成绩是{self.score}")

    def is_passed(self):
        if self.score >= 60 :
            return True
        else:
            return False

# 创建对象（根据模具造饼干）
stu1 = Student("小明",85)
stu2 = Student("小红",55)
# 调用方法
stu1.print_info()
print(stu1.name,"及格了吗?",stu1.is_passed())#stu1.is_passed()不能不加括号，因为需要输出的是返回值而不是这个方法本身
stu2.print_info()
print(stu2.name,"及格了吗?",stu2.is_passed())
# 你的练习：Calculator类
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self,a,b):
        result = a + b
        self.history.append(f"{a}+{b}={result}")
        return result
    
    def subtract(self,a,b):
        result = a - b
        self.history.append(f"{a}-{b}={result}")
        return result

    def show_history(self):
        for h in self.history:
            print(h)

# 测试
calc = Calculator()
print(calc.add(5,3))
print(calc.subtract(91,57))
calc.show_history()

# ========== NumPy 基础 ==========
import numpy as np

#创建数组
a = np.array([1,2,3])
b = np.array([[1,2],[3,4]])

print("a:",a)
print("a的形状:",a.shape)#形状(3,),3是因为数组有3个元素,一维数组：shape 只有一个值（元素个数），不讲行数或列数。numPy 用元组 (3,) 表示一维数组的形状，元组中的逗号是关键，它告诉 Python “这是一个元组”，从而与整数 (3) 区分开来。

print("b:\n",b)
print("b的形状:",b.shape)

# 特殊数组
zeros = np.zeros((2,3)) # 2行3列全0
ones = np.ones((3,)) #  3个1
random = np.random.randn(3) # 3个随机数（标准正态分布）;randn 函数名 “Random Normal” 的缩写，生成标准正态分布随机数

print("zeros:\n",zeros)
print("random:",random)

# 索引和切片
matrix = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

print("第0行:",matrix[0])
print("第0行第1列:",matrix[0,1])
print("第1列:",matrix[:,1])# [2, 5, 8]（所有行的第1列）

# 矩阵运算（线性回归核心！）
X = np.array([[1,2],# 3个样本，2个特征
              [3,4],
              [5,6]])
theta = np.array([0.5,1.0])
# @ 表示矩阵乘法
predictions = X @ theta # NumPy 特殊处理：(n,) 在右边时，自动匹配左边最后一维
print("预测值:",predictions)
#练习
# 1. 创建一个 (4, 3) 的随机矩阵
# 2. 提取第2行（索引1）
# 3. 提取第0列
# 4. 计算矩阵与其转置的乘积
random2 = np.random.randn(4,3)
print("矩阵:\n",random2)
print("形状:", random2.shape)  
print("第二行:",random2[1])
print("第一列:",random2[:,0])
print(f"该矩阵和它的转置的乘积为:\n{random2@random2.T}")









    
