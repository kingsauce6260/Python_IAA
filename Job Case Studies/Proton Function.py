"""
This matrix scaling function is built to scale the columns by a numeric value
from list n input.

Rules to follow:
"We define the matrix-vector product only for the case when the number of
columns in A equals the number of rows in x. So, if A is an m×n matrix
(i.e., with n columns), then the product Ax is defined for n×1 column
vectors x. If we let Ax=b, then b is an m×1 column vector. In other words,
the number of rows in A(which can be anything) determines the number of
rows in the product b." - definition taken from mathinsight
(https://mathinsight.org/matrix_vector_multiplication)

Inputs:
mn:
Should be in either a list or array form. Example: [[1,3,2],[3,2,1]]

n:
Should be in a list form 1xn. Remember n should be length of columns in
matrix mn. Example: [1,4,3]
"""

import numpy as np


def scale_matrix(mn, n):
    np.array(mn)
    if range(len(n)) == range(len(mn[0])):
        for x in range(len(mn[0])):
            for i in range(len(mn)):
                mn[i][x] = mn[i][x] * n[x]
    else:
        print("Correct the input.")
    print(mn)