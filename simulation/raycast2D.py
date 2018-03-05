#! /usr/bin/python3
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import MultiPolygon, box, Point
from shapely.affinity import rotate, translate, scale
from shapely.ops import cascaded_union


parser = argparse.ArgumentParser()
parser.add_argument('--num_rays',
                    help='Number of rays of the LiDAR. (default: 250)',
                    type=int,
                    default=250)
parser.add_argument('--range',
                    help='Maximum range of the LiDAR. (default: 5.0)',
                    type=float,
                    default=5.0)
parser.add_argument('--map_size',
                    help='Size of the map. (default: 10.0)',
                    type=float,
                    default=10.0)
parser.add_argument('--num_items',
                    help='Number of obstacles in the map. (default: 30)',
                    type=int,
                    default=30)
parser.add_argument('--item_size',
                    help='Size of the obstacles. (default: 1.0)',
                    type=float,
                    default=1.0)

args = parser.parse_args()


def raycast(p0, u, q0, v, uxv=None, t_default=np.nan):
    '''
    Input:
    *   p0
        (2)
        Center of the LiDAR.
    *   u
        (n1, 2)
        Displacement vectors for the `n1` LiDAR rays.
    *   q0
        (n2, 2)
        Starting points of other `n2` line segments.
    *   v
        (n2, 2)
        Displacement vectors of other `n2` line segments.
    '''
    d = q0 - p0
    dxv = np.cross(d, v)[None, :]
    dxu = np.vstack([np.cross(d, aa) for aa in u])

    if uxv is None:
        uxv = np.vstack([np.cross(aa, v) for aa in u])

    t = dxv / uxv
    s = dxu / uxv

    t[(s < 0) | (s > 1) | (t < 0) | (t > 1)] = t_default

    return p0 + np.nanmin(t, axis=1, keepdims=True) * u


if args.num_items > 0:
    # generate a random map
    obstacles = []
    w = args.item_size / 2
    for k in range(args.num_items):
        if np.random.rand() > 0.5:
            obstacles.append(box(-w, -w, w, w))
        else:
            obstacles.append(Point(0, 0).buffer(w))

        ss = 1 + np.random.randn() * 0.2
        obstacles[-1] = scale(obstacles[-1], ss, 1 / ss)

        obstacles[-1] = rotate(obstacles[-1], np.random.rand() * 360)

        cc = np.random.rand(2) * args.map_size
        obstacles[-1] = translate(obstacles[-1], *cc)
else:
    obstacles = [box(-args.range,
                 -args.range,
                 args.map_size + args.range,
                 args.map_size + args.range)]

obstacles = cascaded_union(obstacles)
if type(obstacles) is not MultiPolygon:
    obstacles = MultiPolygon([obstacles])

# start (q0) and end (q1) points for all verticies
q0 = np.vstack([np.asarray(obj.exterior.coords)[:-1, :] for obj in obstacles])
q1 = np.vstack([np.asarray(obj.exterior.coords)[1:, :] for obj in obstacles])
v = q1 - q0

fig = plt.figure(1, figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

for obj in obstacles:
    x, y = obj.exterior.coords.xy
    ax.plot(x, y, 'k', lw=1)

ax.axis('scaled')
ax.axis([0, args.map_size, 0, args.map_size])


lines, = ax.plot([], [], lw=0.5)
dot, = ax.plot([], [], 'r+')

# center of the LiDAR
p0 = np.array([args.map_size / 2] * 2)
theta = np.linspace(0, 2 * np.pi, args.num_rays, endpoint=False)
u = args.range * np.vstack([np.cos(theta), np.sin(theta)]).T
uxv = np.vstack([np.cross(aa, v) for aa in u])

def on_move(event):
    if event.xdata is not None and event.ydata is not None:
        p0[:] = [event.xdata, event.ydata]

fig.canvas.mpl_connect('motion_notify_event', on_move)

timestamps = np.empty(10)
def update(i):
    timestamps[1:] = timestamps[:-1]
    timestamps[0] = time.time()

    points = raycast(p0, u, q0, v, uxv=uxv)
    step = max(points.shape[0] // 200, 1)
    lines.set_xdata([[p0[0], pt[0], np.nan] for pt in points[::step]])
    lines.set_ydata([[p0[1], pt[1], np.nan] for pt in points[::step]])

    dot.set_xdata(p0[0])
    dot.set_ydata(p0[1])

    dt = np.mean(timestamps[:-1] - timestamps[1:])

    if i >= timestamps.size:
        print('average fps = {:.4f}'.format(1 / dt))

    return [lines, dot]

ani = FuncAnimation(fig, update, interval=0, blit=True)

fig.tight_layout()
plt.show()
