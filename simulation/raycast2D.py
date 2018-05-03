#! /usr/bin/python3
import time
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from shapely.geometry import MultiPolygon, box, Point
from shapely.affinity import rotate, translate, scale
from shapely.ops import cascaded_union


tol = np.sqrt(np.finfo(float).eps)

parser = argparse.ArgumentParser()
parser.add_argument('--num_rays',
                    help='Number of rays to simulate. (default: 1000)',
                    type=int, default=1000)
parser.add_argument('--range',
                    help='Maximum range of the LiDAR. (default: 5.0)',
                    type=float, default=5.0)
parser.add_argument('--map_size',
                    help='Size of the map. (default: 10.0)',
                    type=float, default=10.0)
parser.add_argument('--num_items',
                    help='Number of obstacles in the map. (default: 30)',
                    type=int, default=30)
parser.add_argument('--item_size',
                    help='Size of the obstacles. (default: 1.0)',
                    type=float, default=1.0)
parser.add_argument('--num_proc',
                    help='Number of threads. If non-positive, use all threads. (default: 0)',
                    type=int, default=0)
parser.add_argument('--color',
                    help='Color of the LiDAR. (default: C0)',
                    type=str, default='C0')

args = parser.parse_args()


assert args.num_proc <= cpu_count()
if args.num_proc <= 0:
    num_proc = cpu_count()
else:
    num_proc = args.num_proc

num_proc = min(num_proc, args.num_rays)


assert args.num_items > 0
# generate a random map
obstacles = []
w = args.item_size / 2
for k in range(args.num_items):
    if np.random.rand() > 0.5:
        obstacles.append(box(-w, -w, w, w))
    else:
        obstacles.append(Point(0, 0).buffer(w, resolution=4))

    ss = 1 + np.random.randn() * 0.2
    obstacles[-1] = scale(obstacles[-1], ss, 1 / ss)

    obstacles[-1] = rotate(obstacles[-1], np.random.rand() * 360)

    cc = np.random.rand(2) * args.map_size
    obstacles[-1] = translate(obstacles[-1], *cc)


obstacles = cascaded_union(obstacles)
if type(obstacles) is not MultiPolygon:
    obstacles = MultiPolygon([obstacles])

# start (q0) and end (q1) points for all vertices
q0 = np.vstack([np.asarray(o.exterior.coords)[:-1, :] for o in obstacles])
q1 = np.vstack([np.asarray(o.exterior.coords)[1:, :] for o in obstacles])
v = q1 - q0

# center of the LiDAR
pt_lidar = np.ones(2) * args.map_size / 2
theta = np.linspace(0, 2 * np.pi, args.num_rays, endpoint=False)
u = args.range * np.vstack([np.cos(theta), np.sin(theta)]).T


fig = plt.figure(1, figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)

for o in obstacles:
    ax.plot(*o.exterior.coords.xy, 'k', lw=0.5)

center, = ax.plot([], [], '+', color=args.color, ms=16)
points, = ax.plot([], [], '.', color=args.color, ms=4)
poly_obsv = patches.Polygon(np.empty([1, 2]), fc=args.color, alpha=0.2)
patch = ax.add_patch(poly_obsv)

ax.axis('scaled')
ax.set_xlim(0, args.map_size)
ax.set_ylim(0, args.map_size)


def raycast(pt_lidar_, u_):
    """
    Input:
    *   pt_lidar_
        (2)
        Center of the LiDAR.
    *   u_
        (n1, 2) numpy.ndarray
        Displacement vectors for the `n1` LiDAR rays.
    """
    d = q0 - pt_lidar_
    dxv = np.cross(d, v)[None, :]
    dxu = np.vstack([np.cross(d, aa) for aa in u_])

    uxv = np.vstack([np.cross(aa, v) for aa in u_])

    uxv[uxv == 0] = tol

    t = dxv / uxv
    s = dxu / uxv

    t[(s < 0) | (s > 1) | (t < 0) | (t > 1)] = 1

    return pt_lidar_ + np.nanmin(t, axis=1, keepdims=True) * u_


def on_move(event):
    if event.xdata is not None and event.ydata is not None:
        pt_lidar[:] = [event.xdata, event.ydata]


fig.canvas.mpl_connect('motion_notify_event', on_move)

pool = Pool(num_proc)
input_args = []
num_pt_per_proc = args.num_rays // num_proc
for k in range(num_proc):
    aa = u[(k * num_pt_per_proc):((k + 1) * num_pt_per_proc)]
    input_args.append((pt_lidar, aa))


timestamps = np.empty(10)


def update(i):
    timestamps[1:] = timestamps[:-1]
    timestamps[0] = time.time()

    xy = np.vstack(pool.starmap(raycast, input_args))

    center.set_xdata(pt_lidar[0])
    center.set_ydata(pt_lidar[1])

    patch.set_xy(xy)

    dist = np.linalg.norm(xy - pt_lidar[None, :], axis=-1)
    idx = dist < (1 - tol) * args.range

    points.set_xdata(xy[idx, 0])
    points.set_ydata(xy[idx, 1])

    dt = np.mean(timestamps[:-1] - timestamps[1:])

    if i >= timestamps.size:
        print('average fps = {:.4f}'.format(1 / dt), end='\r')

    return [patch, center, points]


ani = FuncAnimation(fig, update, interval=0, blit=True)

fig.tight_layout()
plt.show()

pool.close()
