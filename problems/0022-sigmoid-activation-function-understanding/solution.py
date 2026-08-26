import math

def sigmoid(z: float) -> float:
	#Your code here
	if z == 0: 
		return 0.5
	else:
		result = 1 / (1+ math.exp(-z))
	return result