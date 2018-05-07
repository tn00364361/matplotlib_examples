#! /usr/bin/python3
import time
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import shapely.geometry as geo
import shapely.affinity as aff
import shapely.ops as ops


parser = argparse.ArgumentParser()
parser.add_argument('--num_rays',
                    help='Number of rays to simulate. (default: 1200)',
                    type=int, default=1800)
parser.add_argument('--range',
                    help='Maximum range of the LiDAR. (default: 25.0)',
                    type=float, default=25.0)
parser.add_argument('--sigma',
                    help='Standard deviation of the noise. (default: 0.01)',
                    type=float, default=0.01)
parser.add_argument('--map_size',
                    help='Size of the map. (default: 50.0)',
                    type=float, default=50.0)
parser.add_argument('--num_items',
                    help='Number of obstacles in the map. (default: 40)',
                    type=int, default=40)
parser.add_argument('--item_size',
                    help='Size (area) of the obstacles. (default: 25.0)',
                    type=float, default=25.0)
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


def gen_obstacle(seed):
    random.seed(seed)

    if random.rand() > 0.5:
        w = np.sqrt(args.item_size  / 4)
        obstacle = geo.box(-w, -w, w, w)
    else:
        r = np.sqrt(args.item_size / np.pi)
        obstacle = geo.Point(0, 0).buffer(r, resolution=4)

    ss = np.sqrt(np.abs(random.randn()) + 1)
    obstacle = aff.scale(obstacle, ss, 1 / ss)

    obstacle = aff.rotate(obstacle, random.rand() * 360)

    cc = random.rand(2) * args.map_size
    obstacle = aff.translate(obstacle, *cc)

    return obstacle


with Pool(num_proc) as pool:
    seeds = random.randint(2**32 - 1, size=args.num_items)
    obstacles = pool.map(gen_obstacle, seeds)

obstacles = ops.cascaded_union(obstacles)
if type(obstacles) is not geo.MultiPolygon:
    obstacles = geo.MultiPolygon([obstacles])

# start (q0) and end (q1) points for all vertices
q0 = np.vstack([np.asarray(o.exterior.coords)[:-1, :] for o in obstacles])
q1 = np.vstack([np.asarray(o.exterior.coords)[1:, :] for o in obstacles])
v = q1 - q0

# center of the LiDAR
pt_lidar = np.ones(2) * args.map_size / 2
theta = np.linspace(0, 2 * np.pi, args.num_rays, endpoint=False)
u = args.range * np.vstack([np.cos(theta), np.sin(theta)]).T

input_args = []
num_pt_per_proc = args.num_rays // num_proc
for k in range(num_proc):
    aa = u[(k * num_pt_per_proc):((k + 1) * num_pt_per_proc)]
    # aa = u[k::num_proc]
    input_args.append((pt_lidar, aa))


fig = plt.figure(1, figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)

for o in obstacles:
    ax.plot(*o.exterior.coords.xy, color='k', lw=0.5)

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
        (n1, 2)
        Displacement vectors for the `n1` LiDAR rays.
    """
    d = q0 - pt_lidar_[None, :]

    d0, d1 = d[:, 0][None, :], d[:, 1][None, :]
    u0, u1 = u_[:, 0][:, None], u_[:, 1][:, None]
    v0, v1 = v[:, 0][None, :], v[:, 1][None, :]
    dxv = np.cross(d, v)[None, :]
    dxu = d0 * u1 - d1 * u0
    uxv = u0 * v1 - u1 * v0

    t, s = dxv / uxv, dxu / uxv
    t[(s < 0) | (s > 1) | (t < 0) | (t > 1)] = 1
    t_min = np.min(t, axis=1, keepdims=True)

    idx_valid = t_min < 1
    noise = args.sigma / args.range * random.randn(*t_min[t_min < 1].shape)
    t_min[idx_valid] += noise

    return idx_valid, pt_lidar_[None, :] + t_min * u_


def on_move(event):
    if event.xdata is not None and event.ydata is not None:
        pt_lidar[:] = [event.xdata, event.ydata]


fig.canvas.mpl_connect('motion_notify_event', on_move)


pool = Pool(num_proc)
timestamps = np.empty(10)


def update(i):
    center.set_xdata(pt_lidar[0])
    center.set_ydata(pt_lidar[1])

    outputs = pool.starmap(raycast, input_args)
    idx = np.vstack([o[0] for o in outputs]).flatten()
    xy = np.vstack([o[1] for o in outputs])
    patch.set_xy(xy)
    points.set_xdata(xy[idx, 0])
    points.set_ydata(xy[idx, 1])

    timestamps[1:] = timestamps[:-1]
    timestamps[0] = time.time()
    dt = np.mean(timestamps[:-1] - timestamps[1:])

    if i >= timestamps.size:
        print('average fps = {:.4f}'.format(1 / dt), end='\r')

    return patch, center, points


ani = FuncAnimation(fig, update, interval=0, blit=True)

fig.tight_layout()
plt.show()

pool.close()
