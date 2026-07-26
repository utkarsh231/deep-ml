import numpy as np
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:

	a = matrix[0][0]
	b = matrix[0][1]
	c = matrix[1][0]
	d = matrix[1][1]

	trace = a+d 
	det = a*d - b*c

	#b2 - 4ac
	dis = trace**2 - 4 * det

	x_1 = (trace + np.sqrt(dis)) / 2
	x_2 = (trace - np.sqrt(dis)) / 2

	eigenvalues = [x_1, x_2]
	return eigenvalues