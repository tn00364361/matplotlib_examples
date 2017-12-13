import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


'''
Simulate a `Van der Pol oscillator`.

Reference:
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#Two-dimensional_form
'''

mu = 1
omega = 1
def calc_f(t, x):
    x_dot = np.array([x[1], mu * (1 - x[0]**2) * x[1] - omega**2 * x[0]])
    #l = np.linalg.norm(x_dot, axis=0)
    #x_dot = x_dot / l * np.log1p(l)
    return x_dot


def calc_phi(t, x):
    return np.arctan2(x[1], x[0])

# plot the flow in log-scale
axlim = 4
num_cells = 20
temp = np.linspace(-axlim, axlim, num_cells)
X = np.stack(np.meshgrid(temp, temp), axis=0)
F = calc_f(None, X)
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
t_eval = np.arange(0, 20, dt)
num_sim = 20
x = np.empty([2, t_eval.size, num_sim])
for k in range(num_sim):
    x0 = (2 * np.random.rand(2) - 1) * axlim
    #x0 = np.random.randn(2)
    x[..., k] = solve_ivp(calc_f, t_eval[[0, -1]], x0, t_eval=t_eval).y


# animation
lines, dots = [], []
colors = ['C{:d}'.format(i) for i in range(10)]
for k in range(num_sim):
    clr = colors[k % len(colors)]
    lines.append(ax.plot([], [], lw=4, color=clr)[0])
    dots.append(ax.plot([], [], '.', ms=16, color=clr)[0])


def update(i):
    print(t_eval[i])
    i1, i2 = max(0, i - int(0.1 / dt)), i + 1

    for k in range(num_sim):
        lines[k].set_xdata(x[0, i1:i2, k])
        lines[k].set_ydata(x[1, i1:i2, k])

        dots[k].set_xdata(x[0, i, k])
        dots[k].set_ydata(x[1, i, k])

    return lines + dots


if 1:
    ani = FuncAnimation(fig1, update,
        frames=t_eval.size, interval=dt * 1000, blit=True)

if 0:
    dpi = int(1080 / min(fig1.get_size_inches()))
    ani.save('Van der Pol oscillator.mp4', dpi=dpi, fps=1 / dt, bitrate=4096)

plt.show()
