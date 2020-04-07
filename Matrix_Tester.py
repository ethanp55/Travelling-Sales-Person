from MatrixHandler import *
import numpy as np

init_matrix = [[math.inf, 7, 3, 12], [3, math.inf, 6, 14], [5, 8, math.inf, 6], [9, 3, 5, math.inf]]
matrix = np.array(init_matrix)

reduction_cost = Matrix_Handler.reduce_matrix(matrix)

print(reduction_cost)
print(matrix)

