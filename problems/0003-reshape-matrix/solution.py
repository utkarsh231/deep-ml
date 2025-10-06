import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method

    arr = np.array(a)

    total_ele = arr.size

    if total_ele != new_shape[0] * new_shape[1]:
        return []

    reshaped_matrix = arr.reshape(new_shape).tolist()
	return reshaped_matrix