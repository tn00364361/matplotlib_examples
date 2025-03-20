#! /usr/bin/python3
import argparse
from functools import partial
import sys
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


FloarArray = NDArray[np.float32 | np.float64]


def on_scroll(event, order: FloarArray, step_size: float):
    order[:] = np.maximum(order + event.step * step_size, step_size)


def update(i: int, order: FloarArray, line: plt.Line2D):
    xy: FloarArray = line.get_xydata().copy()
    scale = (np.abs(xy[:, 0]) ** order + np.abs(xy[:, 1]) ** order) ** (1 / order)
    line.set_xdata(xy[:, 0] / scale)
    line.set_ydata(xy[:, 1] / scale)

    print(f"order = {order[0]:.4f}", end="\r")

    return [line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--order",
        "-o",
        type=float,
        default=2.0,
        help="Order of the unit ball. (default: 2.0)",
    )
    parser.add_argument(
        "--step",
        "-s",
        type=float,
        default=0.1,
        help="Step size of each scoll event. (default: 0.1)",
    )
    parser.add_argument(
        "--num_points",
        "-n",
        type=int,
        default=1024,
        help="Number of points. (default: 1024)",
    )
    args = parser.parse_args()

    args.num_points = 4 * (args.num_points // 4) + 1

    order = np.array([args.order])
    theta = np.linspace(0, 2 * np.pi, args.num_points)

    fig = plt.figure(1, figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    (line,) = ax.plot(np.cos(theta), np.sin(theta))

    ax.axis("scaled")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True)

    fig.canvas.mpl_connect(
        "scroll_event", partial(on_scroll, order=order, step_size=args.step)
    )
    print("Scroll the mouse wheel to change the order of the unit ball.")
    _ = FuncAnimation(
        fig,
        partial(update, order=order, line=line),
        interval=20,
        blit=True,
        cache_frame_data=False,
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
