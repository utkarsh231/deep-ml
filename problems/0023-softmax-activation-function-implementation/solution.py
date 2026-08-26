import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    d = 0
    ans = []
    m = max(scores)
    for i in scores:
        d += math.exp(i - m)
        
        
    for i in scores:
        ans.append(round(((math.exp(i- m)) / d), 4))
    return ans