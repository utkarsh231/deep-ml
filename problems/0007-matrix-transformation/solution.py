import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:

	A = np.array(A, dtype = float)
	T = np.array(T, dtype = float)
	S = np.array(S, dtype = float)

	if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
		return -1
	
	return np.linalg.inv(T)@ A @ S

# # Manual approach
# 	A = np.array(A, dtype = float)
# 	T = np.array(T, dtype = float)
# 	S = np.array(S, dtype = float)

# 	(sa, sb), (sc, sd) = S
# 	det_s = sa*sd - sb*sc

# 	(ta, tb), (tc, td) = T
# 	det_t = ta*td - tb*tc

# 	if det_s == 0 or det_t == 0:
# 		return -1
	
# 	T_inv = np.array([
# 		[td, -tb],
# 		[-tc, ta]
# 	]) / det_t

# 	transformed_matrix = T_inv @ A @ S

# 	return transformed_matrix

