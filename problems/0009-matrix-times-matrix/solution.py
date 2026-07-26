import numpy as np
def matrixmul(a:list[list[int|float]], b:list[list[int|float]])-> list[list[int|float]]:

    a = np.array(a, dtype = float)
    b = np.array(b, dtype = float)

    # if a.shape != b.shape:
    #     return -1
    if a.shape[1] != b.shape[0]:
        return -1
    
    c = a @ b
    return c