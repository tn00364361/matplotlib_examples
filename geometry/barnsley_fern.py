import numpy as np
import matplotlib.pyplot as plt


'''
Reference:
    https://en.wikipedia.org/wiki/Barnsley_fern
'''

param = np.array([
    [0, 0, 0, 0.16, 0, 0],
    [0.85, 0.04, -0.04, 0.85, 0, 1.60],
    [0.20, -0.26, 0.23, 0.22, 0, 1.60],
    [-0.15, 0.28, 0.26, 0.24, 0, 0.44]
])
prob = np.array([0.01, 0.85, 0.07, 0.07])

def f(x):
    idx = np.random.choice(4, p=prob)
    a, b, c, d, e, f = param[idx, :]

    return np.array([[a, b], [c, d]]) @ x + np.array([e, f])

num_iter = 100000
x = np.zeros([2, num_iter])

for k in range(1, num_iter):
    x[:, k] = f(x[:, k - 1])

fig = plt.figure(1, figsize=(6, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x[0, :], x[1, :], s=2, color='C2', alpha=0.2)
ax.axis('scaled')
ax.set_xlim(-4, 4)
ax.set_ylim(-1, 11)

fig.tight_layout()
plt.show()
