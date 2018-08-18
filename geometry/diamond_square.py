#! /usr/bin/python3
import argparse
import numpy as np
from scipy.ndimage import convolve
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', '-n', type=int, default=10,
                    help='Number iterations. (default: 10)')
parser.add_argument('--decay', type=float, default=2.0,
                    help='Decay factor of the noise level. (default: 2.0)')
parser.add_argument('--sigma', type=float, default=1.0,
                    help='Standard deviation of the Gaussian kernel. (default: 1.0)')
parser.add_argument('--max', type=float, default=1.0,
                    help='Maximum value of the final result. (default: 1.0)')
parser.add_argument('--min', type=float, default=0.0,
                    help='Minimum value of the final result. (default: 0.0)')

args = parser.parse_args()


rv = multivariate_normal(mean=[0, 0], cov=[[args.sigma, 0], [0, args.sigma]])

w = int(np.ceil(4 * args.sigma))
t_grid = np.linspace(-w, w, 2 * w + 1)
t_mid = np.linspace(0.5 - w, w - 0.5, 2 * w)

kc = rv.pdf(np.dstack(np.meshgrid(t_grid, t_grid)))
kc /= np.sum(kc)

km = rv.pdf(np.dstack(np.meshgrid(t_mid, t_mid)))
km /= np.sum(km)

kv = rv.pdf(np.dstack(np.meshgrid(t_grid, t_mid)))
kv /= np.sum(kv)

kh = kv.T


z = np.random.randn(2, 2)
for ni in range(1, args.num_iter + 1):
    z_new = np.random.randn(2**ni + 1, 2**ni + 1)
    z_new /= args.decay**(ni - 1)

    z_new[::2, ::2] += convolve(z, kc)
    z_new[1::2, 1::2] += convolve(z, km)[:-1, :-1]
    z_new[1::2, ::2] += convolve(z, kv)[:-1, :]
    z_new[::2, 1::2] += convolve(z, kh)[:, :-1]

    z = z_new


m = (args.max - args.min) / (z.max() - z.min())
b = args.min - m * z.min()
z = m * z + b


fig = plt.figure(2, figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
cax = ax.imshow(z)
plt.colorbar(cax)

fig.tight_layout()

plt.show()
