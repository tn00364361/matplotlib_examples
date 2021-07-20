#! /usr/bin/python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


parser = argparse.ArgumentParser()
parser.add_argument('--order', '-o', type=float, default=2.0,
                    help='Order of the unit ball. (default: 2.0)')
parser.add_argument('--step', '-s', type=float, default=0.1,
                    help='Step size of each scoll event. (default: 0.1)')
parser.add_argument('--num_pt', '-n', type=int, default=1024,
                    help='Number of points. (default: 1024)')
args = parser.parse_args()

args.num_pt = 4 * (args.num_pt // 4) + 1

p = np.array([args.order])
theta = np.linspace(0, 2 * np.pi, args.num_pt)
x, y = np.cos(theta), np.sin(theta)


fig = plt.figure(1, figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)

line, = ax.plot(x, y)

ax.axis('scaled')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.grid(True)


def on_scroll(event):
    p[:] = max(p + event.step * args.step, args.step)


fig.canvas.mpl_connect('scroll_event', on_scroll)


def update(i):
    scale = (np.abs(x)**p + np.abs(y)**p)**(1 / p)
    line.set_xdata(x / scale)
    line.set_ydata(y / scale)

    print(f'order = {p[0]:.4f}', end='\r')

    return [line]


print('Scroll the mouse wheel to change the order of the unit ball.')
ani = FuncAnimation(fig, update, interval=20, blit=True)

fig.tight_layout()
plt.show()
