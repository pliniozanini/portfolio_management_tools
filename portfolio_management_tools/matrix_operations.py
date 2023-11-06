# matrix_operations.py

import numpy as np

def matrix_multiplication(matrix1, matrix2):
    """
    Perform matrix multiplication
    """
    result = np.dot(matrix1, matrix2)
    return result

def matrix_transpose(matrix):
    """
    Transpose a matrix
    """
    result = np.transpose(matrix)
    return result

# Example usage:
# matrix1 = ...  # Your first matrix
# matrix2 = ...  # Your second matrix
# result = matrix_multiplication(matrix1, matrix2)

# Example usage for matrix transpose:
# matrix = ...  # Your matrix
# transposed_matrix = matrix_transpose(matrix)
