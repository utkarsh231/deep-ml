def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.

	"""
	Conditions: 
		1. Matrix should not be empty 
		2. both matrix and vector should have same dimensions
	"""

	if not a or not a[0]:
		return -1

	num_rows = len(a)
	num_cols = len(a[0])

	if num_rows != len(b):
		return -1

	result = []
	for row in a:
		dot_value = sum(row[i] * b[i] for i in range(num_cols))
		result.append(dot_value)
	
	return result