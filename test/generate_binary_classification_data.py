import pandas as pd
from sklearn.datasets import make_classification

# 生成二分类数据集
num_samples = 10000  # 样本数量
num_features = 5     # 特征数量
X, y = make_classification(n_samples=num_samples, 
                           n_features=num_features, 
                           n_informative=3, 
                           n_redundant=0, 
                           n_classes=2, 
                           random_state=42)

# 将类别转换为'Class_0' 和 'Class_1'
y = ['Class_0' if label == 0 else 'Class_1' for label in y]

# 创建DataFrame
data = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(num_features)])
data.insert(0, 'class', y)  # 将类别作为第一列

# 保存为CSV文件
data.to_csv('binary_classification_data.csv', index=False)

print("二分类模型测试的CSV文件已生成！")
