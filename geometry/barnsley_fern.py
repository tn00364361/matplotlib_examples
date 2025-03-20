#! /usr/bin/python3
import argparse
from functools import partial
import sys
from multiprocessing import Pool, cpu_count
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt


"""
Reference:  https://en.wikipedia.org/wiki/Barnsley_fern
"""


coeffs = np.array(
    [
        [0, 0, 0, 0.16, 0, 0],
        [0.85, 0.04, -0.04, 0.85, 0, 1.60],
        [0.20, -0.26, 0.23, 0.22, 0, 1.60],
        [-0.15, 0.28, 0.26, 0.24, 0, 0.44],
    ],
    dtype=np.float32,
)
prob = np.array([0.01, 0.85, 0.07, 0.07], dtype=np.float32)


def iterate_one_point(seed: int, num_iterations: int):
    rng = np.random.default_rng(seed)
    indices = rng.choice(4, size=num_iterations, p=prob)
    points = np.empty([2, num_iterations + 1], dtype=np.float32)
    points[:, 0] = rng.random(2)
    for k in range(num_iterations):
        A = coeffs[indices[k], :4].reshape([2, 2])
        b = coeffs[indices[k], 4:]
        points[:, k + 1] = A @ points[:, k] + b

    return points[:, 1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_points",
        "-n",
        type=int,
        default=1_000_000,
        help="Number of points. (default: 1000000)",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=float,
        default=0.02,
        help="Resolution of the grids. (default: 0.02)",
    )

    args = parser.parse_args()
    num_proc = cpu_count()
    num_iterations = np.round(args.num_points / num_proc).astype(int)

    with Pool(num_proc) as pool:
        seeds = random.randint(2**32 - 1, size=num_proc)
        xy = np.hstack(
            pool.map(partial(iterate_one_point, num_iterations=num_iterations), seeds)
        )

    x_bin = np.arange(-3, 3, args.resolution)
    y_bin = np.arange(-0.5, 10.5, args.resolution)

    H, _, _ = np.histogram2d(xy[0, :], xy[1, :], bins=[x_bin, y_bin])

    fig = plt.figure(1, figsize=(5, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(
        np.log1p(H).T,
        origin="lower",
        extent=(x_bin.min(), x_bin.max(), y_bin.min(), y_bin.max()),
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
