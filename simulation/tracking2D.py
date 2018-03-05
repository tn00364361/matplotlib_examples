#! /usr/bin/python3
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm, block_diag, norm


parser = argparse.ArgumentParser()
parser.add_argument('--zeta',
                    help='Damping ratio. (default: sqrt(0.5))',
                    type=float,
                    default=np.sqrt(0.5))
parser.add_argument('--tau',
                    help='Time constant in seconds. (default: 0.05)',
                    type=float,
                    default=0.05)
parser.add_argument('--dt',
                    help='Sampling time in seconds. (default: 0.01)',
                    type=float,
                    default=0.01)


args = parser.parse_args()

# second-order continuous-time system (1D)
wn = 1 / args.zeta / args.tau
Ac = np.array([[0, 1], [-wn**2, -2 * args.zeta * wn]])

# second-order discrete-time system (2D)
Ad = expm(block_diag(Ac, Ac) * args.dt)

fig = plt.figure(1, figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

line, = ax.plot([], [], color='C0', lw=4)
dot, = ax.plot([], [], '.', ms=16, color='C0')

ax.axis('scaled')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.grid(True)
fig.tight_layout()

x = np.zeros([4, int(0.1 / args.dt)])
x_des = np.zeros(4)

def on_move(event):
    if event.xdata is not None and event.ydata is not None:
        x_des[[0, 2]] = [event.xdata, event.ydata]

fig.canvas.mpl_connect('motion_notify_event', on_move)

timestamps = np.empty(10)
def update(i):
    timestamps[1:] = timestamps[:-1]
    timestamps[0] = time.time()

    if i > timestamps.size:
        dt = np.mean(timestamps[:-1] - timestamps[1:])
        print('average fps = {:.4f}'.format(1 / dt))

    x[:, 1:] = x[:, :-1]
    x[:, 0] = x_des + Ad @ (x[:, 0] - x_des)

    line.set_xdata(x[0, :])
    line.set_ydata(x[2, :])

    dot.set_xdata(x[0, 0])
    dot.set_ydata(x[2, 0])

    if norm(x[:, 0] - x_des) > 1e-3:
        line.set_color('C3')
        dot.set_color('C3')
    else:
        line.set_color('C0')
        dot.set_color('C0')


    return [dot, line]

ani = FuncAnimation(fig, update, interval=1000 * args.dt, blit=True)


plt.show()
