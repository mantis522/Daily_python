import numpy as np

math = [3, 1]
science = [-2, -2]
history = [-1, 1]

data = np.array([math, science, history])

a = np.cov(data, bias=True)

print(a)