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
                    help='Number of rays to simulate. (default: 1800)',
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
                    help='Number of threads. If non-positive, use all the threads. (default: 4)',
                    type=int, default=4)
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
        w = np.sqrt(args.item_size / 4)
        obstacle = geo.box(-w, -w, w, w)
    else:
        r = np.sqrt(args.item_size / np.pi)
        obstacle = geo.Point(0, 0).buffer(r, resolution=64)
        # print(np.asarray(obstacle.exterior.coords.xy).shape)
        # print(obstacle.exterior)

    ss = np.sqrt(np.abs(random.randn()) + 1)
    obstacle = aff.scale(obstacle, ss, 1 / ss)

    obstacle = aff.rotate(obstacle, random.rand() * 360)

    cc = random.rand(2) * args.map_size
    obstacle = aff.translate(obstacle, *cc)

    return obstacle


with Pool(num_proc) as pool:
    seeds = random.randint(2**32 - 1, size=args.num_items)
    obstacles = pool.map(gen_obstacle, seeds)

obstacles = ops.cascaded_union(obstacles).simplify(args.map_size * 1e-3)
if type(obstacles) is not geo.MultiPolygon:
    obstacles = geo.MultiPolygon([obstacles])


def get_vertices(polygon):
    xy_ext = np.asarray(polygon.exterior.coords.xy)

    xy_int_start, xy_int_end = [], []
    for i in polygon.interiors:
        xy = np.asarray(i.coords.xy)
        xy_int_start.append(xy[:, :-1])
        xy_int_end.append(xy[:, 1:])

    xy_start = np.hstack([xy_ext[:, :-1]] + xy_int_start).T
    xy_end = np.hstack([xy_ext[:, 1:]] + xy_int_end).T
    return xy_start, xy_end


with Pool(num_proc) as pool:
    outputs = pool.map(get_vertices, obstacles)

# start and end points for all vertices
q_start = np.vstack([o[0] for o in outputs])
q_end = np.vstack([o[1] for o in outputs])
v = q_end - q_start

# center of the LiDAR
pt_lidar = random.rand(2) * args.map_size
# while geo.Point(*pt_lidar).within(obstacles):
#     pt_lidar = random.rand(2) * args.map_size

theta = np.linspace(0, 2 * np.pi, args.num_rays, endpoint=False)
u = args.range * np.vstack([np.cos(theta), np.sin(theta)]).T

input_args = []
num_pt_per_proc = np.ceil(args.num_rays / num_proc).astype(int)
for k in range(num_proc):
    aa = u[(k * num_pt_per_proc):((k + 1) * num_pt_per_proc), :]
    input_args.append((pt_lidar, aa))


fig = plt.figure(1, figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)

for o in obstacles:
    xy = np.asarray(o.exterior.coords.xy).T
    ax.add_patch(patches.Polygon(xy, fc=np.ones(3) * 0.5, ec='k', lw=0.5))

    for i in o.interiors:
        xy = np.asarray(i.coords.xy).T
        ax.add_patch(patches.Polygon(xy, fc='w', ec='r', lw=0.5))

center, = ax.plot([], [], '+', color=args.color, ms=16)
points, = ax.plot([], [], '.', color=args.color, ms=4)
poly_obsv = patches.Polygon(np.empty([1, 2]), fc=args.color, alpha=0.5)
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
        (N, 2)
        Displacement vectors for the `N` LiDAR rays.
    """
    d = q_start - pt_lidar_[None, :]

    d0, d1 = d[:, 0], d[:, 1]
    u0, u1 = u_[:, 0], u_[:, 1]
    v0, v1 = v[:, 0], v[:, 1]
    dxv = np.cross(d, v)[None, :]
    dxu = d0[None, :] * u1[:, None] - d1[None, :] * u0[:, None]
    uxv = u0[:, None] * v1[None, :] - u1[:, None] * v0[None, :]

    t, s = dxv / uxv, dxu / uxv
    t[(s < 0) | (s > 1) | (t < 0) | (t > 1)] = 1
    t_min = np.min(t, axis=1)

    idx_valid = t_min < 1
    noise = args.sigma / args.range * random.randn(*t_min[idx_valid].shape)
    t_min[idx_valid] += noise

    return idx_valid, pt_lidar_[None, :] + t_min[:, None] * u_


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
    idx = np.hstack([o[0] for o in outputs])
    xy = np.vstack([o[1] for o in outputs])
    patch.set_xy(xy)

    skip = max(np.round(np.sum(idx) / 1000).astype(int), 1)
    points.set_xdata(xy[idx, 0][::skip])
    points.set_ydata(xy[idx, 1][::skip])

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
