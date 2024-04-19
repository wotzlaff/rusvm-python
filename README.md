# rusvm
A Python interface to [rusvm](https://github.com/wotzlaff/rusvm).

## Installation
```
pip install rusvm
```

## Example
```py
import rusvm
import numpy as np
import matplotlib.pyplot as plt

# generate sample dataset
n = 20
np.random.seed(42)
x = np.random.rand(n)
y = np.sin(2.0 * np.pi * x)

# define parameters for training problem
# regularization parameter
lmbda = 1.0
# scaling parameter
gamma = 10.0

# solve training problem
res = rusvm.solve_smo(
    x=np.sqrt(gamma) * x[:, None],
    y=y,
    params_problem=dict(
        lmbda=lmbda,
        kind='regression',
    ),
    params_smo=dict(
        time_limit=1.0,
    )
)
print(res['opt_status'])
a = np.array(res['a']) / lmbda
a = a[:n] + a[n:]

# generate reference points
xplot = np.linspace(0.0, 1.0, 100)
# evaluate decision function at reference points
k = np.exp(-gamma * (xplot[:, None] - x[None, :]) ** 2)
yplot = k.dot(a) + res['b']

# plot training points and decision function
plt.plot(x, y, 'kx')
plt.plot(xplot, yplot, 'r')
```

## Build
```
maturin develop --release
```