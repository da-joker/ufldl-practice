import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 从 sklearn 的 model_selection 模块中导入 train_test_split 函数。这个函数把数据随机分成训练集（用来训练模型）和测试集（用来评估模型），通常按 80% / 20% 的比例划分。
from sklearn.preprocessing import StandardScaler # 从 sklearn 的 preprocessing 模块导入 StandardScaler 类。这个类用来标准化数据：减去均值除以标准差，让所有特征都在同一个尺度（均值0，标准差1），避免数值大的特征主导模型。
import requests # requests HTTP请求库，用于下载网络数据
import io # io	输入/输出工具，处理内存中的文件
# ========== 1. 数据下载 ==========
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"#url 变量名，用来存储网址。作用：指向一个在线数据文件。这个 URL 是 UCI 机器学习仓库中克利夫兰心脏病数据集的原始 .data 文件地址。文件中每行是一个病人的数据。
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
           'restecg', 'thalach', 'exang', 'oldpeak', 
           'slope', 'ca', 'thal', 'target']#columns	变量名，存放列名

print("正在下载 Heart Disease 数据集...")#作用：在终端显示提示信息，让你知道程序在做什么。
response = requests.get(url)#requests 之前导入的 HTTP 请求库；get(url) 向指定网址发送 GET 请求（下载文件）；response 变量，存放服务器返回的响应。作用：从 url 下载文件，并将结果（包含数据、状态等）保存到 response。下载的内容：就是那个 URL 里的一大堆逗号分隔的数字。每一行是一个病人，最后一列是心脏病诊断结果（0 = 无病，1,2,3,4 = 有病/病情程度）。
data = pd.read_csv(io.StringIO(response.text), header=None, names=columns)#data = 变量，存放读入的表格；pd.read_csv() 把 CSV（逗号分隔值）文件读成 DataFrame（表格）；io.StringIO(...) 把字符串假装成"文件对象"，让 pd.read_csv 可以读；response.text 从下载的 response 中取出文本内容（就是那些数字和逗号）；header=None 原始数据没有表头，不要将第一行当作列名；names=columns	使用上面定义的 columns 列表作为列名。 整体意思：从网络上下载的原始数据只是一堆文本（数字+逗号+换行）。io.StringIO 把它包装成类似文件的格式，pd.read_csv 负责解析成表格，并用 columns 作为表头。
# ========== 2. 数据清洗 ==========
# 处理缺失值（数据集中以 '?' 表示）
data = data.replace('?', np.nan)#'?' 要查找的字符串（缺失值的标记）;nan 指 Not a Number（缺失值）;data.replace(...)在 Pandas 中，replace 方法用于替换 DataFrame 中的某些值。这里是把所有 '?' 字符替换成 np.nan。
data = data.dropna()#data（右）当前的 DataFrame（已经替换了 ? 为 np.nan）; dropna 删除缺失值的方法名; data.dropna() 删除 DataFrame 中包含缺失值 (np.nan) 的行（或列）,默认参数：axis=0：删除行（而不是列）、how='any'：只要某行有一个缺失值，就删除整行、inplace=False：返回新对象，不修改原对象。缺失值处理的其他方法（知识扩展）:删除行 data.dropna()	缺失值很少时; 删除列 data.dropna(axis=1) 某列缺失太多时 ; 填充均值 data.fillna(data.mean())	数值特征; 填充中位数 data.fillna(data.median())	有异常值时 ; 填充众数 data.fillna(data.mode().iloc[0]) 分类特征。
# 二值化目标：0 保持 0，1-4 全部转为 1（有心脏病）
data['target'] = data['target'].apply(lambda x: 1 if int(x) > 0 else 0)#第一部分：data['target']作用：选择 DataFrame 中名为 target 的列。第2部分：.apply(lambda x: ...)作用：对 target 列的每一个元素执行函数。lambda x: 是匿名函数（临时用的小函数），x 代表当前元素的值。第3部分：1 if int(x) > 0 else 0,如果 int(x) > 0（即 x = 1,2,3,4），返回 1,否则（x = 0），返回 0。第4部分：int(x) 的作用原始数据中的 target 可能是整数（0,1,2,3,4）或浮点数（0.0, 1.0, 2.0...）。int(x) 确保转换成整数进行比较，避免浮点精度问题。第5部分：data['target'] = ...将转换后的结果（0 或 1）赋值回 target 列，替换原来的值。
# 转换数值类型
data = data.astype(float)# astype 转换数据类型的方法,replace('?', np.nan) 只替换内容，不改变列类型（仍是 object）。astype(float) 强制将列转为数值类型 float64，使缺失值 np.nan 被正确识别，整列可以参与矩阵乘法、加法等数学运算。没有这一步，后续 X @ w 会因类型不匹配而报错。
print("\n=== 数据集基本信息 ===")#\n 是换行符（转义字符），告诉计算机：在这里换一行。
print(f"样本数: {len(data)}")# f 格式化字符串标记（f-string）：允许在 {} 中写变量或表达式; len(data) 内置函数 len()：返回对象的长度。对 data（DataFrame）来说，返回的是行数，即样本数
print(f"特征数: {len(columns)-1}")#len(columns)	计算 columns 列表的长度。columns 是前面定义好的列名列表，里面有所有列的名称。减去 1 → 去掉最后一列 target（标签列），剩下的就是特征的数量。
print(f"\n类别分布:\n{data['target'].value_counts()}")#  value_counts()  Pandas 方法：统计该列中每个取值出现的次数（默认按频次降序排列）
print(f"\n前5行:\n{data.head()}")#head()	Pandas 方法：返回 DataFrame 的前 5 行（默认）,比如head(4) 就是前4行。
# ========== 3. EDA：可视化 ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 10))# fig 变量名：整个画布（Figure），axes 变量名：子图数组（Axes），subplots 函数名：创建子图网格，“2, 2”参数：2 行、2 列（共 4 个子图），figsize=(12, 10)	画布大小：12 英寸宽，10 英寸高。
# 年龄分布
axes[0,0].hist(data[data['target']==0]['age'], bins=15, alpha=0.6, label='No Disease')# hist 直方图方法。data	DataFrame（包含所有数据）。data['target']==0	布尔条件：target 列等于 0 的行（没有心脏病）。data[ ... ] 用布尔索引选出满足条件的行。['age'] 取这些行的 age 列（年龄数据）。, 分隔参数。bins=15 将年龄范围分成 15 个等宽的区间（柱子数量）。alpha=0.6 透明度（0 完全透明，1 完全不透明）。 label='No Disease'	图例标签，用于区分两条曲线。 在 axes[0,0] 子图上画直方图，数据是 没有心脏病（target=0）的样本的年龄，分成 15 个柱子，透明度 0.6，图例标签为 'No Disease'。注意：这里 data[data['target']==0]['age'] 会返回一个 Pandas Series，里面全是整数或浮点数。
axes[0,0].hist(data[data['target']==1]['age'], bins=15, alpha=0.6, label='Disease')#先筛选行（data[布尔条件]），再选列（['age']）。这是 Pandas 常见的链式索引。
axes[0,0].set_xlabel('Age')#axes[0,0] 左上角子图,set_xlabel	设置 X 轴标签的方法.作用：给 X 轴加上文字说明“Age”（年龄）。
axes[0,0].set_ylabel('Count')# 作用：给 Y 轴加上文字说明“Count”（人数频次）。
axes[0,0].legend()# legend 显示图例的方法。作用：自动使用之前 hist 中指定的 label 参数（'No Disease' 和 'Disease'）生成图例框，显示在子图合适位置。 如果不加 legend() 会怎样？不会显示图例，无法区分两个直方图分别代表哪类人。
axes[0,0].set_title('Age Distribution by Target')
# 胆固醇 vs 最大心率-----在 2×2 子图的右上角绘制散点图，展示胆固醇与最大心率的关系，点的颜色表示有无心脏病（蓝=无病，红=有病），同时添加坐标轴标签、标题和颜色条，便于直观分析两类人群在这两个指标上的分布差异。
scatter = axes[0,1].scatter(data['chol'], data['thalach'], 
                           c=data['target'], cmap='coolwarm', alpha=0.6)#scatter	变量名，用来存储 scatter 方法返回的对象（后面加颜色条要用）。[0,1]	索引：第 0 行、第 1 列（右上角子图）。scatter 散点图方法。c= 参数：指定每个点的颜色值，data['target'] 取 'target' 列（0 或 1）作为颜色映射的值，cmap='coolwarm'	颜色映射表（colormap），将数值（0,1）映射为蓝色到红色。 作用：在右上角子图画出散点图，横轴胆固醇，纵轴最大心率，点颜色根据心脏病状态（0/1）显示为蓝色/红色。
axes[0,1].set_xlabel('Cholesterol')#作用：给 X 轴加上标签 Cholesterol。
axes[0,1].set_ylabel('Max Heart Rate')#作用：给 Y 轴加上标签 Max Heart Rate。
axes[0,1].set_title('Cholesterol vs Max Heart Rate')
plt.colorbar(scatter, ax=axes[0,1])#colorbar 创建颜色条的方法，scatter 前面存储的散点图对象（告诉颜色条要用它的颜色映射），作用：在图的右侧添加一个颜色条，显示蓝色代表 0（无心脏病），红色代表 1（有心脏病）。如果不把 scatter 赋值给变量，能直接 plt.colorbar() 吗？可以，使用 plt.colorbar(ax=axes[0,1]) 也能自动找到当前子图上的散点图对象，但为了清晰，显式传递 scatter 是更推荐的做法（尤其当子图中有多个图形时）。
# 相关性热力图
corr = data.corr()#corr	变量名，用于存储计算出的相关系数矩阵；corr() 方法名，计算 DataFrame 中所有数值列的两两相关系数（默认皮尔逊相关系数）。作用：生成一个方形的相关系数矩阵，大小 = (特征数, 特征数)，每个元素表示两个特征之间的线性相关程度：值为 +1：完全正相关，值为 -1：完全负相关，值为 0：无线性相关
im = axes[1,0].imshow(corr, cmap='RdBu', vmin=-1, vmax=1)#变量名，存储 imshow 返回的 AxesImage 对象（方便后面添加颜色条）；imshow 方法名，将二维数组显示为图像（热力图）；在左下角子图上显示一个热力图，每个单元格的颜色表示该位置相关系数的强弱和方向（蓝→负相关，红→正相关，白→弱相关）。
axes[1,0].set_xticks(range(len(corr.columns)))#set_xticks 设置 X 轴主刻度位置的方法；range(len(corr.columns)) 生成 0 到 (特征数-1) 的整数序列。作用：将 X 轴刻度放在整数位置 0, 1, 2, … 上，与热力图的列索引一一对应。
axes[1,0].set_yticks(range(len(corr.columns)))
axes[1,0].set_xticklabels(corr.columns, rotation=90, fontsize=8)#set_xticklabels 替换 X 轴刻度标签文本的方法
axes[1,0].set_yticklabels(corr.columns, fontsize=8)
axes[1,0].set_title('Feature Correlation Matrix')
plt.colorbar(im, ax=axes[1,0])#作用：在左下角子图的右侧添加一个垂直颜色条，显示颜色与相关系数值的对应关系（从 -1（蓝）到 1（红））。
# 目标分布-------这段代码在右下角子图上绘制目标变量（target）的条形图，统计有无心脏病的样本数量，添加坐标轴标签、标题，并将 X 轴刻度替换为“No Disease”和“Disease”，用不同颜色区分两类，用于观察类别平衡情况。
data['target'].value_counts().plot(kind='bar', ax=axes[1,1], color=['skyblue', 'salmon'])
axes[1,1].set_xlabel('Target (0=No, 1=Yes)')
axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Target Distribution')
axes[1,1].set_xticklabels(['No Disease', 'Disease'], rotation=0)

plt.tight_layout()
plt.savefig('E:/AI/ufldl_practice/heart_eda.png', dpi=150)
plt.show()
print("\nEDA 图表已保存")
#上面是一个数据探索分析（EDA） 流程
# ========== 4. UFLDL 风格的逻辑回归实现 ==========
class UFLDL_LogisticRegression:
    """
    基于 UFLDL Tutorial 的逻辑回归实现
    使用批量梯度下降(Batch Gradient Descent)
    """
    def __init__(self, learning_rate=0.1, n_iterations=1000, #__init__ 存储超参数（学习率、迭代次数、正则化系数）
                 regularization=0.01):#learning_rate 0.1 学习率 α，控制梯度下降的步长；n_iterations	1000 训练迭代次数（遍历整个数据集的次数）；regularization 0.01 L2 正则化系数 λ，用于防止过拟合
        self.lr = learning_rate#self.lr、self.n_iter 等成为实例属性，所有方法（比如将来的 fit、predict）都可以通过 self.lr 访问到学习率。如果没有 self，这些变量就只是 __init__ 里的局部变量，无法在其他方法中使用。
        self.n_iter = n_iterations#传入的迭代次数，存储最大迭代次数
        self.lambda_reg = regularization  # L2 正则化（UFLDL 优化章节内容）
        self.theta = None#模型参数（权重向量 + 截距）。训练开始前为 None，训练时会被初始化为合适的形状。
        self.loss_history = []#列表，记录每次迭代后的损失值，可用于绘制损失曲线。
    def _sigmoid(self, z):#_sigmoid 私有方法，计算概率转换函数。
        """Sigmoid / Logistic 函数"""
        # 防止溢出
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))#np.clip(z, -250, 250)	将 z 中的每个元素限制在区间 [-250, 250] 内（小于 -250 的变成 -250，大于 250 的变成 250）。
    def _add_intercept(self, X):#_add_intercept	方法名。单下划线开头表示这是一个“约定私有方法”，仅供类内部使用，外部不应直接调用。代表类的实例对象，通过它可以访问实例的其他属性和方法（如 self.lr, self.theta）。
        """添加截距项(x_0 = 1)"""
        return np.c_[np.ones((X.shape[0], 1)), X]#c_ 是 NumPy 中一个类似于索引对象的东西，用于按列（column-wise）拼接两个数组。它可以把多个数组“并排”放在一起，要求行数相同。等价于 np.hstack，但语法更简洁。np.ones 生成一个全 1 的数组。参数 (X.shape[0], 1)：X.shape 返回一个元组 (m, n)，m 是样本数，n 是特征数。X.shape[0] 就是样本数 m。(X.shape[0], 1) 表示生成一个 (m, 1) 的二维数组，即 m 行、1 列，每个元素都是 1。, X – 要拼接的第二个数组要把原始特征矩阵 X（形状 (m, n)）放在全 1 列的右侧。np.c_[左侧数组, 右侧数组]
    def compute_cost(self, X, y):#compute_cost	方法名，表示计算代价（损失）。
        """
        UFLDL 代价函数（带 L2 正则化）:
        J(θ) = -1/m * Σ[y*log(h) + (1-y)*log(1-h)] + λ/2m * Σθ²
        """
        m = len(y)#len(y) 获取标签向量 y 的长度（即样本总数）。
        h = self._sigmoid(X @ self.theta)
         # 对数损失
        cost = -1/m * np.sum(y * np.log(h + 1e-5) + 
                            (1-y) * np.log(1 - h + 1e-5))#加上一个极小偏移（如 1e-5）后，log 的参数永远不会 ≤ 0，从而保证数值稳定性。
        
        # L2 正则化（不包含 theta_0）
        reg = self.lambda_reg / (2*m) * np.sum(self.theta[1:]**2)
        return cost + reg
    def fit(self, X, y, verbose=True):#X	特征矩阵，形状 (m, n_features)，不含截距项（本方法内部会调用 _add_intercept 添加）。y	标签向量，形状 (m,)，取值为 0 或 1。verbose	布尔值，为 True 时打印训练过程中的损失。
        """
        训练模型 - 批量梯度下降
        梯度: ∇J = 1/m * X^T(h-y) + λ/m * θ
        """
        X = self._add_intercept(X)#self._add_intercept(X) 在原始特征左侧添加一列全 1，得到 X 形状 (m, n_original + 1)。
        m, n = X.shape
        # 初始化参数（小随机值，UFLDL 推荐）
        self.theta = np.random.randn(n) * 0.001
        for i in range(self.n_iter):#重复执行循环体内的代码 self.n_iter 次。
            # 前向传播
            z = X @ self.theta# X 特征矩阵（已包含截距列），形状 (m, n)。self.theta	参数向量，形状 (n,)。z[i] = X[i,0]*theta[0] + X[i,1]*theta[1] + ... + X[i,n-1]*theta[n-1]。结果 z 是形状 (m,) 的一维数组。
            h = self._sigmoid(z)#作用：对 z 的每个元素应用 sigmoid 函数，得到概率 h，形状 (m,)，取值范围 (0,1)。
            # 计算梯度
            gradient = 1/m * (X.T @ (h - y))#X.T X 的转置，形状 (n, m)。无正则化时，损失函数对 theta 的梯度为 (1/m) * X^T (h - y)。
            # 正则化梯度（theta_0 不正则化）
            gradient[1:] += (self.lambda_reg / m) * self.theta[1:]#作用：完成带 L2 正则化的总梯度计算。
            # 参数更新
            self.theta -= self.lr * gradient#作用：沿负梯度方向更新参数：θ_new = θ_old - α * ∇J。
            # 记录损失
            loss = self.compute_cost(X, y)
            self.loss_history.append(loss)
            if verbose and i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.6f}")#每 100 次迭代，如果 verbose 为 True，则打印一次当前损失，便于观察收敛情况。verbose	实例方法的参数（布尔值），决定是否打印信息。and	逻辑与，两个条件同时为真才执行。i % 100	取模运算，i 除以 100 的余数。== 0	等于 0，判断是否每 100 次迭代。f"..."	格式化字符串，{i} 和 {loss:.6f} 被变量的值替换。{i}	插入当前迭代次数。{loss:.6f}	插入损失值，保留 6 位小数。
        
        return self
    def predict_proba(self, X):
        """预测概率 P(y=1|x)"""
        X = self._add_intercept(X)#self._add_intercept	类内部定义的私有方法，用于在特征矩阵左侧添加一列全 1。
        return self._sigmoid(X @ self.theta)#最终返回：形状为 (m,) 的一维数组，第 i 个元素是第 i 个样本属于类别 1 的概率。
    def predict(self, X, threshold=0.5):#predict 方法通过调用 predict_proba 获得概率，然后与用户指定的阈值（默认 0.5）比较，将 True/False 转换为 1/0 整数数组，从而输出样本的最终类别标签。
        """预测类别"""
        return (self.predict_proba(X) >= threshold).astype(int)
    def accuracy(self, X, y):#accuracy 方法通过比较模型预测结果与真实标签，使用 NumPy 的 mean 函数计算正确标签的比例，从而评估分类模型的整体准确率。
        """计算准确率"""
        pred = self.predict(X)
        return np.mean(pred == y)

# ========== 5. 数据预处理与训练 ==========
# 分离特征与标签
X = data.drop('target', axis=1).values#.drop() 是 Pandas DataFrame 的方法，用于删除指定的行或列。第一个参数 'target'：要删除的列名（或行索引）。这里删除名为 'target' 的列。第二个参数 axis=1：axis=0 表示删除行（默认）。axis=1 表示删除列。返回值：一个新的 DataFrame，包含除 'target' 列以外的所有列（即所有特征列）。
y = data['target'].values#.values属性：Pandas DataFrame 和 Series 都有的 .values 属性。作用：将 DataFrame（或 Series）转换为 NumPy 数组（ndarray）。
# 划分训练集/测试集---------train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 将特征 X 和标签 y 随机划分为训练集（80%）和测试集（20%），固定随机种子确保结果可重复，并使用分层抽样保持原始数据中各类别的比例，防止类别不平衡导致的评估偏差。
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)#train_test_split 返回一个包含 4 个对象的元组（tuple），Python 会自动解包给等号左边的 4 个变量。test_size=0.2含义：指定测试集的比例（或绝对数量）。值：0.2 表示 20% 的数据用于测试，剩余 80% 用于训练。random_state=42含义：随机数生成器的种子（seed）。作用：固定划分的随机性，使得每次运行代码时，分割的结果完全相同。42 是任意选择的常用种子（可以是 0, 1, 42, 123 等）。如果不设置，每次运行划分结果都会不同，影响实验可复现性。stratify=y含义：分层抽样，根据标签 y 的分布进行划分。
# 特征标准化（UFLDL 标准做法：零均值，单位方差）
scaler = StandardScaler()#StandardScaler 对训练集进行 fit_transform 以学习每个特征的均值和标准差并标准化，对测试集只用 transform 应用相同的变换，从而保证训练和测试数据同分布，符合 UFLDL 提倡的零均值、单位方差标准化，有助于提高梯度下降收敛速度和正则化效果。
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\n=== 开始训练逻辑回归 ===")
model = UFLDL_LogisticRegression(
    learning_rate=0.5, 
    n_iterations=1000,
    regularization=0.1
)
model.fit(X_train_scaled, y_train)#model = UFLDL_LogisticRegression(learning_rate=0.5, n_iterations=1000, regularization=0.1) 创建了一个逻辑回归模型实例，设置了学习率、迭代次数和正则化系数；然后 model.fit(X_train_scaled, y_train) 使用标准化后的训练数据执行批量梯度下降训练，更新模型参数 self.theta 并记录损失历史，最终得到一个可用的分类器。
# ========== 6. 模型评估 ==========
train_acc = model.accuracy(X_train_scaled, y_train)#调用 model.accuracy，传入训练集数据，模型内部先用 predict 得到预测标签，再与真实标签比较，计算出正确预测的比例，并将该准确率赋值给 train_acc。示例：如果训练集有 237 个样本，模型预测正确了 220 个，则 train_acc = 220/237 ≈ 0.9283。
test_acc = model.accuracy(X_test_scaled, y_test)#作用：计算模型在未见过的测试集上的准确率，并存入 test_acc。注意：测试集从未参与训练，它的准确率更能反映模型的泛化能力。
print(f"\n=== 训练结果 ===")
print(f"训练集准确率: {train_acc:.4f}")#.4f	格式说明：浮点数，保留 4 位小数
print(f"测试集准确率: {test_acc:.4f}")
# 详细指标
from sklearn.metrics import classification_report, confusion_matrix#从 sklearn.metrics 模块中导入两个函数，以便后续使用，无需写完整的 sklearn.metrics.confusion_matrix。
y_pred = model.predict(X_test_scaled)#作用：调用 model.predict 得到测试集每个样本的预测类别（0 或 1），存入 y_pred。y_pred 的形状为 (m_test,)，与 y_test 相同。
print(f"\n混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")
print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])}")
# ========== 7. 训练过程可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))#ig, axes = plt.subplots(1, 2, figsize=(14, 5)) 创建了一个宽度 14 英寸、高度 5 英寸的画布，并在其中生成 1 行 2 列的子图网格，返回整个画布对象 fig 和子图数组 axes，用于后续可视化训练过程中的损失曲线、准确率变化等。
# 损失曲线
axes[0].plot(model.loss_history, color='blue', linewidth=2)#作用：在左侧子图上画出横坐标为迭代次数（索引顺序 0,1,2,…,len-1），纵坐标为损失值的折线。model.loss_history 列表的长度等于 self.n_iter（如 1000），因此横轴自动从 0 到 999。
axes[0].set_xlabel('Iteration')#作用：在子图下方添加文字标签“Iteration”，说明横坐标的含义。
axes[0].set_ylabel('Cost J(θ)')#作用：在子图左侧添加标签“Cost J(θ)”，说明纵坐标是损失值。
axes[0].set_title('UFLDL Logistic Regression: Training Loss')
axes[0].grid(True, alpha=0.3)#True	第一个参数，布尔值，表示开启网格线（False 则关闭）。grid	方法名，用于显示网格线。
# 特征重要性（theta 权重）----这段代码在右侧子图上绘制水平条形图，展示逻辑回归模型学习到的参数 θ（包括截距和各个特征的权重），并用颜色区分正负权重（红正蓝负），添加零线以直观比较特征对分类决策的影响方向和强度。
feature_names = ['Intercept'] + list(data.drop('target', axis=1).columns)
colors = ['gray'] + ['red' if t > 0 else 'blue' for t in model.theta[1:]]
axes[1].barh(feature_names, model.theta, color=colors, alpha=0.7)
axes[1].set_xlabel('Weight Value')
axes[1].set_title('Learned Parameters (θ)')
axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()#tight_layout	函数名，用于自动调整子图之间的间距和边距，防止标签、标题、刻度等元素重叠或超出画布。
plt.savefig('E:/AI/ufldl_practice/logistic_regression_results.png', dpi=150)#savefig 函数名，将当前图形保存为文件。dpi=150 关键字参数，设置输出图片的分辨率为 150 每英寸点数（dots per inch）。数值越高图像越清晰，文件也越大。默认值通常为 100。
plt.show()


