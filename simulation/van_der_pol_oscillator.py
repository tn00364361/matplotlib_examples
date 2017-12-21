import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


'''
Simulate a `Van der Pol oscillator`.

Reference:
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#Two-dimensional_form
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mu',
                    help='Damping coefficient. (default: 1.0)',
                    type=float,
                    default=1.0)
parser.add_argument('--omega',
                    help='Natural frequency in rad/s. (default: 1.0)',
                    type=float,
                    default=1.0)
parser.add_argument('--num_sim',
                    help='Number of simulations. (default: 25)',
                    type=int,
                    default=25)
parser.add_argument('--t_span',
                    help='Duration for the simulation in seconds. (default: 20.0)',
                    type=float,
                    default=20.0)
parser.add_argument('--no-animation',
                    dest='animate',
                    help='Disable animation.',
                    action='store_false')
parser.add_argument('--save',
                    dest='save_video',
                    help='Save the animation as `Van der Pol oscillator.mp4`',
                    action='store_true')
parser.add_argument('--log-quiver',
                    dest='log_quiver',
                    help='Plot the flow in log-scale.',
                    action='store_true')
parser.set_defaults(animate=True, log_quiver=False, save_video=False)

args = parser.parse_args()


def calc_f(t, x):
    x_dot = np.array([x[1], args.mu * (1 - x[0]**2) * x[1] - args.omega**2 * x[0]])
    return x_dot


def calc_phi(t, x):
    return np.arctan2(x[1], x[0])

axlim = 4
num_cells = 20
temp = np.linspace(-axlim, axlim, num_cells)
X = np.stack(np.meshgrid(temp, temp), axis=0)
F = calc_f(None, X)

if args.log_quiver:
    V = np.linalg.norm(F, axis=0).reshape([1, num_cells, num_cells])
    F = F / V * np.log1p(V)

fig1 = plt.figure(1, figsize=(6, 6))
ax = fig1.add_subplot(1, 1, 1)
ax.quiver(X[0, ...], X[1, ...], F[0, ...], F[1, ...], pivot='mid')

ax.axis('scaled')
ax.axis([-axlim, axlim, -axlim, axlim])

fig1.tight_layout()

# approximate the limit cycle (steady-state trajectory)
dt = 0.02
t_eval = np.arange(0, 1000, dt)

event = calc_phi
event.terminal = False
event.direction = -1
sol = solve_ivp(calc_f, t_eval[[0, -1]], [2, 0], t_eval=t_eval, events=event)
t_event = sol.t_events[0]
last_cycle = (t_eval >= t_event[-2] - dt) & (t_eval <= t_event[-1] + dt)
x_ss = sol.y[:, last_cycle]
t_ss = t_eval[last_cycle]

ax.plot(x_ss[0, :], x_ss[1, :], color=np.ones(3) * 0.5)


# solve the ODE given `num_sim` initial conditions
t_eval = np.arange(0, args.t_span, dt)
x = np.empty([2, t_eval.size, args.num_sim])
for k in range(args.num_sim):
    x_init = (2 * np.random.rand(2) - 1) * axlim
    x[..., k] = solve_ivp(calc_f, t_eval[[0, -1]], x_init, t_eval=t_eval).y


# animation
lines, dots = [], []
colors = ['C{:d}'.format(i) for i in range(10)]
for k in range(len(colors)):
    clr = colors[k % len(colors)]
    lines.append(ax.plot([], [], lw=4, color=clr)[0])
    dots.append(ax.plot([], [], '.', ms=16, color=clr)[0])


def update(i):
    print('t = {:.4f}'.format(t_eval[i]))
    i1, i2 = max(0, i - int(0.1 / dt)), i + 1

    for k in range(len(colors)):
        idx = ((np.arange(args.num_sim) + k) % len(colors)) == 0

        data = np.empty([i2 - i1 + 1, np.sum(idx)])
        data[-1, :] = np.nan

        data[:(i2 - i1), :] = x[0, ...][i1:i2, idx]
        lines[k].set_xdata(data.copy().T)
        data[:(i2 - i1), :] = x[1, ...][i1:i2, idx]
        lines[k].set_ydata(data.copy().T)

        dots[k].set_xdata(x[0, i, idx])
        dots[k].set_ydata(x[1, i, idx])

    return lines + dots


if args.animate:
    ani = FuncAnimation(fig1,
                        update,
                        frames=t_eval.size,
                        interval=dt * 1000,
                        blit=True)

    if args.save_video:
        ani.save('Van der Pol oscillator.mp4',
                 dpi=int(1080 / fig1.get_size_inches()[1]),
                 fps=1 / dt,
                 bitrate=4096)

plt.show()
