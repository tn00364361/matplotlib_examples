#! /usr/bin/python3
import argparse
import time
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy.linalg import expm, toeplitz
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''
Solving 1D Fokker-Planck equation using spectral method with Fourier basis.
'''

parser = argparse.ArgumentParser()

parser.add_argument('--duration', type=float, default=5.0,
                    help='Duration of the simulation. (default: 5.0)')
parser.add_argument('--dt', type=float, default=0.02,
                    help='Sampling time in seconds. (default: 0.02)')
parser.add_argument('--Nx', type=int, default=256,
                    help='Number of samples in x. (default: 256)')
parser.add_argument('--width', type=float, default=4.0,
                    help='Half of the width in x. (default: 4.0)')

args = parser.parse_args()


x = np.linspace(-1, 1, args.Nx) * args.width

# flow
f = -x * (x - args.width / 2) * (x + args.width / 2) / (1 + x**2)

# diffusion (must be non-negative everywhere)
g = np.zeros_like(x) + 0.1

assert np.all(g >= 0)

F, G = toeplitz(fft(f)), toeplitz(fft(g))

omega = fftfreq(args.Nx, d=x[1] - x[0]) * 2 * np.pi
Dx, Dxx = 1j * omega, -omega**2

Ac = (Dxx[:, None] * G - Dx[:, None] * F) / args.Nx
Ad = expm(Ac * args.dt)

fig = plt.figure(1, figsize=(6, 6))
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(x, f, label='f(x)')
ax1.plot(x, g, label='g(x)')
ax1.axhline(0, color='k', lw=0.5)
ax1.set_xlim(x.min(), x.max())
ax1.legend()

ax2 = fig.add_subplot(2, 1, 2)
line, = ax2.plot(x, np.nan * x, label='p(x)')
ax2.axhline(0, lw=0.5, color='k')
ax2.set_xlim(x.min(), x.max())
ax2.set_ylim(-0.1, 2)
ax2.set_xlabel('x')
ax2.legend()

fig.tight_layout()

p_hat = np.empty(args.Nx, dtype=complex)
timestamps = np.zeros(10)
Nt = np.round(args.duration / args.dt).astype(int)


def update(i):
    if i:
        p_hat[:] = np.dot(Ad, p_hat)
        p = ifft(p_hat).real
    else:
        mu = np.random.randn() * args.width / 6
        sigma = (0.5 + np.random.rand()) * args.width / 6
        p = np.exp(-0.5 * (x - mu)**2 / sigma**2) / np.sqrt(2 * np.pi) / sigma
        # p = np.zeros_like(x)
        # p[np.random.randint(args.Nx)] = 1
        p_hat[:] = fft(p)

    line.set_ydata(p)

    timestamps[1:] = timestamps[:-1]
    timestamps[0] = time.time()
    dt = np.mean(timestamps[:-1] - timestamps[1:])
    print('[{:6d}/{:6d}] Average FPS = {:.4f}'.format(i, Nt, 1 / dt), end='\r')

    return [line]


ani = FuncAnimation(fig, update, frames=Nt, interval=args.dt * 1000, blit=True)

plt.show()
