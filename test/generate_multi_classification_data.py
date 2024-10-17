import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# 生成三分类数据集
num_samples = 100000  # 样本数量
num_features = 5      # 特征数量
n_classes = 3        # 类别数量
X, y = make_classification(n_samples=num_samples, 
                           n_features=num_features, 
                           n_informative=3, 
                           n_redundant=0, 
                           n_classes=n_classes, 
                           random_state=42)

# 将类别转换为'Class_0', 'Class_1', 'Class_2'
y = [f'Class_{label}' for label in y]

# 创建DataFrame
data = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(num_features)])
data.insert(0, 'class', y)  # 将类别作为第一列

# 保存为CSV文件
data.to_csv('multi_classification_data.csv', index=False)

print("多分类模型测试的CSV文件已生成！")

# 数据集分割：70%训练集，30%测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用Logistic回归作为分类器
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs')
classifier.fit(X_train, y_train)

# 预测
y_pred = classifier.predict(X_test)

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(pd.DataFrame(conf_matrix, 
                   columns=[f'Class_{i}' for i in range(n_classes)], 
                   index=[f'Class_{i}' for i in range(n_classes)]))

# 准确率（Accuracy）
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.5f}")

# 错误率（Error Rate）
error_rate = 1 - accuracy
print(f"Error Rate: {error_rate:.5f}")

# 精确率（Precision）
precision = precision_score(y_test, y_pred, average=None, labels=[f'Class_{i}' for i in range(n_classes)])
precision_df = pd.DataFrame([precision], columns=[f'Class_{i}' for i in range(n_classes)], index=['precision'])
print("\nPrecision:")
print(precision_df)

# 召回率（Recall）
recall = recall_score(y_test, y_pred, average=None, labels=[f'Class_{i}' for i in range(n_classes)])
recall_df = pd.DataFrame([recall], columns=[f'Class_{i}' for i in range(n_classes)], index=['recall'])
print("\nRecall:")
print(recall_df)
