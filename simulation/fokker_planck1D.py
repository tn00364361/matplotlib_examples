#! /usr/bin/python3
import argparse
import time
import numpy as np
from scipy import fft
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
f = 0.5 * (x**2 / (1 + x**2) + 1)

# diffusion (must be non-negative everywhere)
g = np.full_like(x, 5e-3)

assert np.all(g >= 0)

F = toeplitz(fft.fft(f, norm='ortho'))
G = toeplitz(fft.fft(g, norm='ortho'))

omega = fft.fftfreq(args.Nx, d=x[1] - x[0]) * 2 * np.pi
Dx, Dxx = 1j * omega, -omega**2

Ac = (Dxx[:, None] * G - Dx[:, None] * F) / np.sqrt(args.Nx)
Ad = expm(Ac * args.dt)

fig = plt.figure(1, figsize=(6, 6))
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(x, f, label=r'$f(t, x)$')
ax1.plot(x, g, label=r'$g(t, x)$')
ax1.axhline(0, color='k', lw=0.5)
ax1.set_xlim(x.min(), x.max())
ax1.legend()

ax2 = fig.add_subplot(2, 1, 2)
line_0, = ax2.plot(x, np.nan * x, 'r--', label=r'$p(0, x)$', lw=1)
line_t, = ax2.plot(x, np.nan * x, label=r'$p(t, x)$')
ax2.axhline(0, lw=0.5, color='k')
ax2.set_xlim(x.min(), x.max())
ax2.set_ylim(-0.5, 5)
ax2.set_xlabel(r'$x$')
ax2.legend()

fig.tight_layout()

p_hat = np.empty(args.Nx, dtype=complex)
timestamps = np.zeros(10)
Nt = np.round(args.duration / args.dt).astype(int)


def update(i):
    if i:
        p_hat[:] = np.dot(Ad, p_hat)
        p = fft.irfft(p_hat[:(1 + args.Nx // 2)], norm='ortho')
    else:
        p = 1 + 0.6 * np.random.rand(x.size)
        p[np.abs(x + args.width / 2) >= 1] = 0

        p_hat[:] = fft.fft(p, norm='ortho')

        line_0.set_ydata(p)

    line_t.set_ydata(p)

    timestamps[1:] = timestamps[:-1]
    timestamps[0] = time.time()
    dt = np.mean(timestamps[:-1] - timestamps[1:])
    print(f'[{i:6d}/{Nt:6d}] Average FPS = {1 / dt:.4f}', end='\r')

    return [line_t, line_0]


ani = FuncAnimation(fig, update, frames=Nt, interval=args.dt * 1000, blit=True)

plt.show()
