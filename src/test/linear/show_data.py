import pandas as pd
import matplotlib.pyplot as plt

# 给定关于收入和受教育年限之间的关系的坐标集
# 从csv文件中读取
data = pd.read_csv('./income.csv')

# 以Education为x轴，Income为y轴绘制散点图
plt.scatter(data.Education, data.Income)

# 打印（结果显示x和y呈线性回归关系）
plt.show()
