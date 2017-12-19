import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import argparse


'''
Reference:
    https://en.wikipedia.org/wiki/Lorenz_system
'''

parser = argparse.ArgumentParser()
parser.add_argument('--no-animate',
                    dest='animate',
                    help='Disable animation.',
                    action='store_false')
parser.add_argument('--save',
                    dest='save_video',
                    help='Save the animation as `Lorenz attractor.mp4`',
                    action='store_true')
parser.add_argument('--duration',
                    help='Duration for the simulation in seconds. (default: 50.0)',
                    type=float)
parser.set_defaults(animate=True, save_video=False, duration=50.0)

args = parser.parse_args()


rho, sigma, beta = 28, 10, 8 / 3
def calc_f(t, x):
    x_dot = np.array([
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2]
        ])
    return x_dot


# simulate with a very high sampling rate
dt = 0.001
t_eval = np.arange(0, args.duration, dt)

x_init = np.random.randn(3) + [0, 0, rho - 1]
x1 = solve_ivp(calc_f, t_eval[[0, -1]], x_init, t_eval=t_eval).y

x_init += np.random.randn(3) * 1e-8
x2 = solve_ivp(calc_f, t_eval[[0, -1]], x_init, t_eval=t_eval).y

fig = plt.figure(1, figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
for x in [x1, x2]:
    ax.plot(x[0, :], x[1, :], x[2, :], color='k', lw=0.5, alpha=0.5)

ax.view_init(elev=15, azim=-45)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_zlim(0, 50)

fig.tight_layout()

if args.animate:
    from matplotlib.animation import FuncAnimation

    lines, dots = [], []
    for clr in ['C0', 'C3']:
        lines.append(ax.plot([], [], [], color=clr, lw=2)[0])
        dots.append(ax.plot([], [], [], '.', color=clr, ms=12)[0])

    # set the frame rate of the animation to be 50 Hz
    step = int(0.02 / dt)

    def update(i):
        i1, i2 = max(0, step * i - int(0.1 / dt)), step * i + 1
        #print(t_eval[i2 - 1])

        for k, x in enumerate([x1, x2]):
            lines[k].set_data(x[:2, i1:i2])
            lines[k].set_3d_properties(x[2, i1:i2])

            dots[k].set_data(x[:2, i2 - 1])
            dots[k].set_3d_properties(x[2, i2 - 1])

        #fig.canvas.draw()

        return lines + dots

    ani = FuncAnimation(fig, update,
                        frames=t_eval.size // step, blit=True, interval=dt * step * 1000)

    if args.save_video:
        dpi = int(1080 / fig.get_size_inches()[1])
        ani.save('Lorenz attractor.mp4', dpi=dpi, fps=1 / (dt * step), bitrate=4096)


plt.show()