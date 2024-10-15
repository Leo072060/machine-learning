import numpy as np
import pandas as pd
from scipy.linalg import lu

# 随机生成一个 n x n 的矩阵
matrix_size = 10
# random_matrix = np.random.rand(matrix_size, matrix_size)
random_matrix = np.random.randint(0.1, 50, size=(matrix_size, matrix_size))
# 将矩阵保存到CSV文件中
csv_file_path = 'random_matrix.csv'
pd.DataFrame(random_matrix).to_csv(csv_file_path, index=False, header=False)

print(random_matrix)







# 行列式
determinant = np.linalg.det(random_matrix)
print("Determinant:", determinant)

# LU分解
P, L, U = lu(random_matrix)
print("\nL Matrix:\n", L)
print("\nU Matrix:\n", U)




