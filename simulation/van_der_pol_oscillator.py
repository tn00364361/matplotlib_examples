#! /usr/bin/python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point


'''
Simulate a `Van der Pol oscillator`.

Reference:
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#Two-dimensional_form
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mu', type=float, default=1.0,
                    help='Damping coefficient. (default: 1.0)')
parser.add_argument('--omega', type=float, default=1.0,
                    help='Natural frequency in rad/s. (default: 1.0)')
parser.add_argument('--num_sim', type=int, default=25,
                    help='Number of simulations. (default: 25)')
parser.add_argument('--duration', type=float, default=20.0,
                    help='Duration for the simulation in seconds. (default: 20.0)')
parser.add_argument('--no-animation', dest='animate', action='store_false',
                    help='Disable animation.')
parser.add_argument('--save', dest='save_video', action='store_true',
                    help='Save the animation as `Van der Pol oscillator.mp4`')
parser.add_argument('--log-quiver', dest='log_quiver', action='store_true',
                    help='Plot the flow in log-scale.')
parser.set_defaults(animate=True, log_quiver=False, save_video=False)

args = parser.parse_args()


def calc_f(t, x, mu=args.mu, omega=args.omega):
    x = x.reshape([2, -1])
    x_dot = np.empty_like(x)
    x_dot[0, :] = x[1, :]
    x_dot[1, :] = mu * (1 - x[0, :]**2) * x[1, :] - omega**2 * x[0, :]
    return x_dot.flatten()


axlim = 4
num_cells = 20
temp = np.linspace(-axlim, axlim, num_cells)
X = np.stack(np.meshgrid(temp, temp), axis=0)
F = calc_f(None, X).reshape([2, num_cells, num_cells])

if args.log_quiver:
    V = np.linalg.norm(F, axis=0, keepdims=True)
    F = F / V * np.log1p(V)

fig1 = plt.figure(1, figsize=(6, 6))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.quiver(X[0, ...], X[1, ...], F[0, ...], F[1, ...], pivot='mid')

ax1.axis('scaled')
ax1.set_xlim(-axlim, axlim)
ax1.set_ylim(-axlim, axlim)

fig1.tight_layout()

# Numerically calculate the limit cycle (steady-state trajectory)
dt = 0.001
t_eval = np.arange(0, 100 / args.omega, dt)


def event(t, x):
    return x[1]


event.terminal = True
event.direction = -1

eps = np.finfo(np.float64).eps


def poincare_map(d):
    sol = solve_ivp(calc_f, t_eval[[0, -1]], [d, -eps], t_eval=t_eval, events=event)
    return sol.y[0, -1]


d_fp = fixed_point(poincare_map, 2)


sol = solve_ivp(calc_f, t_eval[[0, -1]], [d_fp, -eps], t_eval=t_eval, events=event)
x_ss = sol.y
x_ss[:, 0] = 0.5 * (x_ss[:, 0] + x_ss[:, -1])
x_ss[:, -1] = x_ss[:, 0]
print('Fixed point at x = {}'.format(x_ss[:, 0]))
print('Period = {}'.format(sol.t_events[0][0]))

ax1.plot(x_ss[0, :], x_ss[1, :], color=np.ones(3) * 0.5)


# solve the ODE given `num_sim` initial conditions
t_eval = np.arange(0, args.duration, dt)
x_init = (2 * np.random.rand(2, args.num_sim) - 1) * axlim
sol = solve_ivp(calc_f, [0, args.duration], x_init.flatten(), t_eval=t_eval)
x = sol.y.reshape([2, args.num_sim, -1])

# animation
lines, dots = [], []
colors = ['C{:d}'.format(i) for i in range(10)]
for k in range(len(colors)):
    clr = colors[k % len(colors)]
    lines.append(ax1.plot([], [], lw=4, color=clr)[0])
    dots.append(ax1.plot([], [], '.', ms=16, color=clr)[0])


step = int(0.02 / dt)


def update(i):
    # print('t = {:.4f}'.format(t_eval[step * i]))
    i1, i2 = max(0, step * i - int(0.1 / dt)), step * i + 1

    for k in range(len(colors)):
        idx = np.mod(np.arange(args.num_sim) + k, len(colors)) == 0

        lines[k].set_xdata([np.append(xx, np.nan) for xx in x[0, idx, i1:i2]])
        lines[k].set_ydata([np.append(xx, np.nan) for xx in x[1, idx, i1:i2]])

        dots[k].set_xdata(x[0, idx, step * i])
        dots[k].set_ydata(x[1, idx, step * i])

    return lines + dots


if args.animate:
    ani = FuncAnimation(
        fig1,
        update,
        frames=t_eval.size // step,
        interval=dt * step * 1000,
        blit=True
    )

    if args.save_video:
        ani.save(
            'Van der Pol oscillator.mp4',
            dpi=int(1080 / fig1.get_size_inches()[1]),
            fps=1 / dt,
            bitrate=4096
        )

plt.show()
