import numpy as np
import smorust

n, nft = 10, 2

np.random.seed(0)
x = np.random.rand(n, nft)
t = x.sum(axis=1)
y = np.where(t > t.mean(), 1.0, -1.0)
smorust.solve_classification(x, y)
