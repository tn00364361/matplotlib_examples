import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


'''
Reference:
    https://en.wikipedia.org/wiki/Lorenz_system
'''

rho, sigma, beta = 28, 10, 8 / 3
def calc_f(t, x):
    x_dot = np.array([
                sigma * (x[1] - x[0]),
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - beta * x[2]
            ])
    return x_dot

dt = 0.002
t_eval = np.arange(0, 50, dt)

x_init = np.random.randn(3) + [0, 0, rho - 1]
x1 = solve_ivp(calc_f, t_eval[[0, -1]], x_init, t_eval=t_eval).y

x_init += np.random.randn(3) * 1e-6
x2 = solve_ivp(calc_f, t_eval[[0, -1]], x_init, t_eval=t_eval).y

fig = plt.figure(1, figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot(x1[0, :], x1[1, :], x1[2, :], color=np.ones(3) * 0.5, lw=0.5)
ax.plot(x2[0, :], x2[1, :], x2[2, :], color=np.ones(3) * 0.5, lw=0.5)

ax.view_init(elev=15, azim=-45)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_zlim(0, 50)

fig.tight_layout()

lines, dots = [], []
for clr in ['C0', 'C3']:
    lines.append(ax.plot([], [], [], color=clr, lw=2)[0])
    dots.append(ax.plot([], [], [], 'o', color=clr, ms=4)[0])


step = int(0.02 / dt)

def update(i):
    i1, i2 = max(0, step * i - int(0.1 / dt)), step * i + 1
    print(t_eval[i2 - 1])

    for k, x in enumerate([x1, x2]):
        lines[k].set_data(x[:2, i1:i2])
        lines[k].set_3d_properties(x[2, i1:i2])

        dots[k].set_data(x[:2, i2 - 1])
        dots[k].set_3d_properties(x[2, i2 - 1])

    ax.view_init(30, 0.5 * i)
    fig.canvas.draw()

    return lines + dots


if 1:
    ani = FuncAnimation(fig, update,
        frames=t_eval.size // step, blit=True, interval=dt * step * 1000)

if 1:
    dpi = int(1080 / min(fig.get_size_inches()))
    ani.save('Lorenz attractor.mp4', dpi=dpi, fps=1 / (dt * step), bitrate=4096)


plt.show()
