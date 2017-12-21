import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point, minimize


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

axlim = 4
num_cells = 20
temp = np.linspace(-axlim, axlim, num_cells)
X = np.stack(np.meshgrid(temp, temp), axis=0)
F = calc_f(None, X)

if args.log_quiver:
    V = np.linalg.norm(F, axis=0).reshape([1, num_cells, num_cells])
    F = F / V * np.log1p(V)

fig1 = plt.figure(1, figsize=(6, 6))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.quiver(X[0, ...], X[1, ...], F[0, ...], F[1, ...], pivot='mid')

ax1.axis('scaled')
ax1.axis([-axlim, axlim, -axlim, axlim])

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
    lines.append(ax1.plot([], [], lw=4, color=clr)[0])
    dots.append(ax1.plot([], [], '.', ms=16, color=clr)[0])


step = int(0.02 / dt)
def update(i):
    #print('t = {:.4f}'.format(t_eval[step * i]))
    i1, i2 = max(0, step * i - int(0.1 / dt)), step * i + 1

    for k in range(len(colors)):
        idx = ((np.arange(args.num_sim) + k) % len(colors)) == 0

        lines[k].set_xdata([np.append(arr, np.nan) for arr in x[0, i1:i2, idx]])
        lines[k].set_ydata([np.append(arr, np.nan) for arr in x[1, i1:i2, idx]])

        dots[k].set_xdata(x[0, step * i, idx])
        dots[k].set_ydata(x[1, step * i, idx])

    return lines + dots


if args.animate:
    ani = FuncAnimation(fig1,
                        update,
                        frames=t_eval.size // step,
                        interval=dt * step * 1000,
                        blit=True)

    if args.save_video:
        ani.save('Van der Pol oscillator.mp4',
                 dpi=int(1080 / fig1.get_size_inches()[1]),
                 fps=1 / dt,
                 bitrate=4096)

plt.show()
