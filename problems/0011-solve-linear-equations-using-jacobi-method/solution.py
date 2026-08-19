import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:

	x = np.zeros(len(b))
	for _ in range(n):
		new_x = np.zeros(len(b))
		for i in range(len(b)):
			total = 0

			for j in range(len(b)):
				if i != j:
					total += A[i][j] * x[j]

			new_x[i] = (b[i] - total) / A[i][i]
		x = new_x
	return x