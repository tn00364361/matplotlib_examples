#! /usr/bin/python3
import argparse
import time
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy.stats import norm
from scipy.linalg import expm, toeplitz
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


parser = argparse.ArgumentParser()

parser.add_argument('--duration',
                    help='Duration of the simulation. (default: 5.0)',
                    type=float,
                    default=5.0)
parser.add_argument('--dt',
                    help='Sampling time in seconds. (default: 0.02)',
                    type=float,
                    default=0.02)


args = parser.parse_args()

Nt = int(args.duration // args.dt) + 1
t = np.linspace(0, args.duration, Nt)

Nx = 256
x = np.linspace(-1, 1, Nx) * 2 * np.pi

# flow
f = np.sin(x)

# diffusion (must be non-negative everywhere)
g = np.zeros_like(x) + 0.2

F = toeplitz(fft(f))
G = toeplitz(fft(g))

omega = fftfreq(Nx, d=x[1] - x[0]) * 2 * np.pi
Dx = 1j * omega
Dxx = -omega**2

Ac = Dxx.reshape([-1, 1]) * G - Dx.reshape([-1, 1]) * F
Ac /= Nx

Ad = expm(Ac * args.dt)

fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
ax.axhline(0, lw=0.5, color='k')
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel('x')
ax.set_ylabel('p')

fig.tight_layout()

line, = ax.plot(x, np.nan * x)

P = np.empty(Nx, dtype=complex)
timestamps = np.zeros(10)


def update(i):

    if i == 0:
        p0 = norm.pdf(x, np.random.randn(), 0.5 + np.random.rand())
        P[:] = fft(p0)

    P[:] = np.dot(Ad, P)
    p = ifft(P).real

    timestamps[1:] = timestamps[:-1]
    timestamps[0] = time.time()
    freq = 1 / np.mean(timestamps[:-1] - timestamps[1:])
    print('\raverage FPS = {:.4f}'.format(freq), end='')

    line.set_ydata(p)

    return [line]


ani = FuncAnimation(fig, update, frames=Nt, interval=args.dt * 1000, blit=True)

plt.show()
