import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round

	X = np.array(X, dtype=float)
	y = np.array(y, dtype=float)	

	theta = (np.transpose(X)) @ X
	theta = np.linalg.inv(theta)

	theta = theta @ np.transpose(X) @ y

	theta = np.round(theta, 4)

	return theta.tolist()