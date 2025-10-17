def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:

	n_cols = len(matrix[0])
	if mode == "row":
		return [sum(r) / n_cols for r in matrix]

	else:
		return [sum(col) / len(matrix) for col in zip(*matrix)]