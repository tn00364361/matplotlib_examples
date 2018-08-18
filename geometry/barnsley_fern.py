#! /usr/bin/python3
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt


'''
Reference:
    https://en.wikipedia.org/wiki/Barnsley_fern
'''

parser = argparse.ArgumentParser()
parser.add_argument('--num_pt', '-n', type=int, default=1000000,
                    help='Number of points. (default: 1000000)')
parser.add_argument('--resolution', '-r', type=float, default=0.02,
                    help='Resolution of the grids. (default: 0.02)')

args = parser.parse_args()


coeffs = np.array([[0, 0, 0, 0.16, 0, 0],
                   [0.85, 0.04, -0.04, 0.85, 0, 1.60],
                   [0.20, -0.26, 0.23, 0.22, 0, 1.60],
                   [-0.15, 0.28, 0.26, 0.24, 0, 0.44]])
prob = np.array([0.01, 0.85, 0.07, 0.07])


def f(xy):
    idx = random.choice(4, p=prob)
    A = coeffs[idx, :4].reshape([2, 2])
    b = coeffs[idx, 4:]
    return np.dot(A, xy) + b


num_proc = cpu_count()
num_iter = np.round(args.num_pt / num_proc).astype(int) + 1


def iterate(seed):
    random.seed(seed)

    points = np.zeros([2, num_iter])
    for k in range(1, num_iter):
        points[:, k] = f(points[:, k - 1])

    return points[:, 1:]


with Pool(num_proc) as pool:
    seeds = random.randint(2**32 - 1, size=num_proc)
    xy = np.hstack(pool.map(iterate, seeds))

x_bin = np.arange(-3, 3, args.resolution)
y_bin = np.arange(-0.5, 10.5, args.resolution)

H = np.histogram2d(xy[0, :], xy[1, :], bins=[x_bin, y_bin])[0]

fig = plt.figure(1, figsize=(5, 8))
ax = fig.add_subplot(1, 1, 1)

ax.imshow(np.log1p(H).T,
          origin='lower',
          extent=(x_bin.min(), x_bin.max(), y_bin.min(), y_bin.max()))

fig.tight_layout()
plt.show()
