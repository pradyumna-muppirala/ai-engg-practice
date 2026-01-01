import numpy as np

"""
Mathematical Operations and Linear Algebra Practice Module
This module demonstrates fundamental matrix operations and advanced linear algebra
concepts using NumPy. It covers:
1. Basic Matrix Operations:
    - Addition and subtraction of matrices
    - Scalar multiplication
    - Element-wise (Hadamard) multiplication
    - Matrix multiplication using dot product
2. Special Matrices:
    - Identity matrices (multiplicative neutral element)
    - Zero matrices (additive neutral element)
    - Diagonal matrices
    - Block diagonal matrices
3. Matrix-Vector Operations:
    - Matrix-vector multiplication with 1D and 2D vectors
    - Identity matrix properties in multiplication
4. Matrix Properties:
    - Determinant calculation (indicates matrix invertibility)
    - Matrix inversion (solving linear systems)
5. Advanced Linear Algebra:
    - Eigenvalue and Eigenvector decomposition
    - Singular Value Decomposition (SVD)
Eigenvalues and Eigenvectors:
# Eigenvalues and eigenvectors are fundamental in understanding matrix transformations.
# For a matrix A, if v is an eigenvector and λ (lambda) is an eigenvalue,
# then A*v = λ*v. This means the matrix transformation only scales the eigenvector
# by the eigenvalue without changing its direction. Eigenvalues reveal the scaling
# factors of the transformation, while eigenvectors show the directions that remain
# unchanged. They are crucial in: dimensionality reduction (PCA), stability analysis,
# quantum mechanics, vibration analysis, and finding principal components in data.
"""

# Define two sample 2x2 matrices
matrix_a = np.array([[1, 2],
                     [3, 4]])

matrix_b = np.array([[5, 6],
                     [7, 8]])

# Matrix Addition
addition = matrix_a + matrix_b
print("Matrix Addition:\n", addition)

# Matrix Subtraction
subtraction = matrix_a - matrix_b
print("\nMatrix Subtraction:\n", subtraction)

# Scalar Multiplication
scalar = 2
scalar_mult = matrix_a * scalar
print("\nScalar Multiplication (multiply by 2):\n", scalar_mult)

# Element-wise Multiplication (Hadamard product)
element_wise = matrix_a * matrix_b
print("\nElement-wise Multiplication:\n", element_wise)

# Matrix Multiplication
matrix_mult = np.dot(matrix_a, matrix_b)
print("\nMatrix Multiplication:\n", matrix_mult)

# Identity Matrix (5x5)
identity_matrix = np.eye(5)
print("\nIdentity Matrix (5x5):\n", identity_matrix)

# Zero Matrix (5x5)
zero_matrix = np.zeros((5, 5))
print("\nZero Matrix (5x5):\n", zero_matrix)

# Diagonal Matrix (5x5)
diagonal_matrix = np.diag([1, 2, 3, 4, 5])
print("\nDiagonal Matrix (5x5):\n", diagonal_matrix)

#Hands-on exercises
# Sample 3x3 matrix and 3D vector
matrix_3x3 = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

vector_3d = np.array([1, 2, 3])

# Matrix-Vector Multiplication
result = np.dot(matrix_3x3, vector_3d)
print("\nMatrix-Vector Multiplication:\n", result)


vector_3d_T = vector_3d.reshape((3,1))
print (vector_3d_T)
result = np.dot(matrix_3x3, vector_3d_T)
print("\nMatrix-Vector Multiplication:\n", result)

# Create a 3x3 identity matrix
identity_3x3 = np.eye(3)

# Create a regular 3x3 matrix
regular_matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

# Multiply the identity matrix with the regular matrix
result_identity_mult = np.dot(identity_3x3, regular_matrix)
print("\nIdentity Matrix (3x3) * Regular Matrix (3x3):\n", result_identity_mult)

# Sample 2x2 matrix
sample_matrix = np.array([[4, 7],
                          [2, 6]])

# Compute determinant
determinant = np.linalg.det(sample_matrix)
print("\nDeterminant of 2x2 matrix:\n", determinant)

# Compute inverse
inverse_matrix = np.linalg.inv(sample_matrix)
print("\nInverse of 2x2 matrix:\n", inverse_matrix)

# Create a 2x2 block diagonal matrix
block_diag_matrix = np.block([[sample_matrix, np.zeros((2, 2))],
                              [np.zeros((2, 2)), sample_matrix]])
print("\n2x2 Block Diagonal Matrix:\n", block_diag_matrix)

#Engen Values and Eigen Vectors
eigenValues, eigenVectors = np.linalg.eig(identity_3x3)
print("Eigen values =>", eigenValues)
print("Eigen Vectors =>", eigenVectors)

# Matrix decomposition
U, S, Vt = np.linalg.svd(identity_3x3)
print("Left singular Vector =>\n", U)
print("Singular Vector =>\n", S)
print("Right Singular Vector Transport =>\n", Vt)

# Reconstruct the original matrix from SVD components
# Formula: A = U * Σ * V^T, where Σ is a diagonal matrix of singular values
reconstructed_matrix = U @ np.diag(S) @ Vt
print("Reconstructed Matrix from SVD:\n", reconstructed_matrix)

# Verify reconstruction is close to original
print("\nOriginal Matrix:\n", identity_3x3)
print("\nReconstruction Error:\n", np.allclose(identity_3x3, reconstructed_matrix))