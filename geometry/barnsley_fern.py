#! /usr/bin/python3
from multiprocessing import Pool, cpu_count
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt


'''
Reference:
    https://en.wikipedia.org/wiki/Barnsley_fern
'''

coeffs = np.array([[0, 0, 0, 0.16, 0, 0],
                   [0.85, 0.04, -0.04, 0.85, 0, 1.60],
                   [0.20, -0.26, 0.23, 0.22, 0, 1.60],
                   [-0.15, 0.28, 0.26, 0.24, 0, 0.44]])
prob = np.array([0.01, 0.85, 0.07, 0.07])


def f(x):
    idx = random.choice(4, p=prob)
    return np.dot(coeffs[idx, :4].reshape([2, 2]), x) + coeffs[idx, 4:]


num_points = 1000000
num_proc = cpu_count()
num_iter = np.round(num_points / num_proc).astype(int) + 1


def iterate(seed):
    random.seed(seed)

    x = np.zeros([2, num_iter])
    for k in range(1, num_iter):
        x[:, k] = f(x[:, k - 1])

    return x[:, 1:]


with Pool(num_proc) as pool:
    seeds = random.randint(2**32 - 1, size=num_proc)
    x = np.hstack(pool.map(iterate, seeds))


fig = plt.figure(1, figsize=(6, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x[0, :], x[1, :], '.', ms=2, alpha=min(1e4 / num_points, 1))
ax.axis('scaled')
ax.set_xlim(-4, 4)
ax.set_ylim(-1, 11)

fig.tight_layout()
plt.show()
