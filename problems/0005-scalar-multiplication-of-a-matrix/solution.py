def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:

    return [[scalar * x for x in row] for row in matrix]

    result = 0
    for row in matrix:
        for x in row:
            result = scalar * x

	return result