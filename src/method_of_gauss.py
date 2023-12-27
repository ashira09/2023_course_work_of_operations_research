from fractions import Fraction
import numpy as np

from fractions import Fraction
import numpy as np

def is_singular(A, b):
    matrix = np.array([float(A[i][j]) for i in range(len(b)) for j in range(len(b))]).reshape(len(b), len(b))
    extended_matrix = np.array([np.append(string, float(b[idx])) for idx, string in enumerate(matrix)])
    rank_of_matrix = np.linalg.matrix_rank(matrix)
    rank_of_extended_matrix = np.linalg.matrix_rank(extended_matrix)
    if rank_of_matrix == rank_of_extended_matrix and rank_of_matrix == len(b):
        return False
    else:
        return True

def gaussian(A, b):
    reshaped_b = b.reshape((len(b), 1))
    A = np.hstack((A, reshaped_b))
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            A[j] -= A[i] * A[j][i] / A[i][i]  
    x = np.array([0] * len(b), dtype=Fraction)
    i = len(A) - 1
    while i >= 0:
        x[i] = (A[i][-1] - sum(x * A[i][0:-1])) / A[i][i] 
        i -= 1  
    return x
    

def solve_linear_system_of_equtations(A, B, solve):
    if is_singular(A, B):
        return False
    solve.extend(list(gaussian(A, B)))
    return True
