import pandas as pd
import numpy as np

# 设置随机种子以便结果可重复
np.random.seed(42)

# 生成数据
num_samples = 100
feature1 = np.random.rand(num_samples) * 100  # 特征1
feature2 = np.random.rand(num_samples) * 50   # 特征2
noise = np.random.randn(num_samples) * 10      # 随机噪声

# 生成目标变量（假设目标是特征1和特征2的线性组合加上噪声）
target = 3 * feature1 + 2 * feature2 + noise

# 创建DataFrame
data = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'target': target
})

# 保存为CSV文件
data.to_csv('regression_dataset.csv', index=False)

print("CSV文件已生成！")
