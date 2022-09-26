import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

""" 实现线性回归函数的求解 """

# 从csv文件中读取
data = pd.read_csv('./income.csv')

# 定义X轴，Y轴
x, y = data.Education, data.Income

# 建立线性模型
model = tf.keras.Sequential()

# 定义输出数据（y）维度是1，输入数据（x）的维度是1
# 最终构建出：y = ax + b
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))

# 优化方法为adam，损失函数为均方差函数（min square error）
model.compile(optimizer='adam', loss='mse')

# 开始训练，对所有数据训练50000次，寻找a的最小值
model.fit(x, y, epochs=50000)

# 分别绘制原来的图像和预测的函数
plt.scatter(x, y)
plt.scatter(x, model.predict(x))

# 显示结果（可以看出拟合效果非常不错）
plt.show()
