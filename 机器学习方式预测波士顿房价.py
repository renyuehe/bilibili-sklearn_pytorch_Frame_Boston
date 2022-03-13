from sklearn import datasets # 数据集

from sklearn.model_selection import train_test_split # 划分数据

from sklearn import preprocessing # 数据预处理

from sklearn import linear_model # 线性模型
from sklearn import kernel_ridge

from sklearn.metrics import r2_score # 评估

# 1、加载数据
boston_data = datasets.load_boston()#导入波士顿房价数据

# 2、划分训练集与测试机
x, y = boston_data.data, boston_data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 3、数据预处理
scalar = preprocessing.StandardScaler().fit(x) # 标准化
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)

# 4、创建模型
kr = linear_model.LinearRegression() # 线性回归模型
# kr = linear_model.Ridge(0.2) # 岭回归
# kr = linear_model.Lasso(0.2) # Lasso回归
# kr = linear_model.ElasticNet(0.3,0.3) # 弹性网络
# kr = kernel_ridge.KernelRidge(kernel="rbf", gamma=0.1) # 核岭回归模型   优先选用此模型

# 5、模型拟合
kr.fit(x_train, y_train)

# 6、预测
y_pred = kr.predict(x_test)

# 7、.评估
r2score = r2_score(y_test, y_pred)
print(r2score)
