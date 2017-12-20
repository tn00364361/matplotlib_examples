import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import MultiPolygon, box
from shapely.affinity import rotate, translate, scale
from shapely.ops import cascaded_union


parser = argparse.ArgumentParser()
parser.add_argument('--num_rays',
                    help='Number of rays of the LiDAR. (default: 200)',
                    type=int,
                    default=200)
parser.add_argument('--range',
                    help='Maximum range of the LiDAR. (default: 10.0)',
                    type=float,
                    default=10.0)
parser.add_argument('--map_size',
                    help='Size of the map. (default: 20.0)',
                    type=float,
                    default=20.0)
parser.add_argument('--num_items',
                    help='Number of obstacles in the map. (default: 25)',
                    type=int,
                    default=25)
parser.add_argument('--item_size',
                    help='Size of the obstacles. (default: 2.0)',
                    type=float,
                    default=2.0)

args = parser.parse_args()


def raycast(p0, u, q0, v):
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
    pt_lidar = np.empty_like(u)
    d = q0 - p0
    dxv = np.cross(d, v)
    for k in range(pt_lidar.shape[0]):
        dxu = np.cross(d, u[k, :])
        uxv = np.cross(u[k, :], v)
        s = dxu / uxv
        t = dxv / uxv

        valid = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)

        if np.any(valid):
            pt_lidar[k, :] = p0 + t[valid].min() * u[k, :]
        else:
            pt_lidar[k, :] = p0 + u[k, :]

    return pt_lidar


# generate a random map
boxes = []
size = args.item_size
for k in range(args.num_items):
    tt = np.random.rand() * 360
    cc = np.random.rand(2) * args.map_size
    ss = 1 + np.random.randn() * 0.2

    boxes.append(box(-size / 2, -size / 2, size / 2, size / 2))
    boxes[-1] = translate(rotate(scale(boxes[-1], ss, 1 / ss), tt), *cc)

polygons = cascaded_union(boxes)
if type(polygons) is not MultiPolygon:
    polygons = MultiPolygon([polygons])

# start (q0) and end (q1) points for all verticies
q0 = np.vstack([np.asarray(p.exterior.coords)[:-1, :] for p in polygons])
q1 = np.vstack([np.asarray(p.exterior.coords)[1:, :] for p in polygons])
v = q1 - q0

fig = plt.figure(1, figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

for p in polygons:
    x, y = p.exterior.coords.xy
    ax.plot(x, y, 'k', lw=1)

ax.axis('scaled')
ax.axis([0, args.map_size, 0, args.map_size])


lines, = ax.plot([], [], lw=0.2)

# center of the LiDAR
p0 = np.array([args.map_size / 2] * 2)
theta = np.linspace(0, 2 * np.pi, args.num_rays, endpoint=False)
u = args.range * np.vstack([np.cos(theta), np.sin(theta)]).T

def on_move(event):
    if event.xdata is not None and event.ydata is not None:
        p0[:] = [event.xdata, event.ydata]

fig.canvas.mpl_connect('motion_notify_event', on_move)

timestamps = np.empty(10)
def update(i):
    timestamps[:-1] = timestamps[1:]
    timestamps[-1] = time.time()

    pt_lidar = raycast(p0, u, q0, v)
    lines.set_xdata([[p0[0], pt[0]] for pt in pt_lidar])
    lines.set_ydata([[p0[1], pt[1]] for pt in pt_lidar])

    fps = 1 / np.mean(timestamps[1:] - timestamps[:-1])

    if i >= timestamps.size:
        print('average fps = {:.4f}'.format(fps))

    return [lines]

ani = FuncAnimation(fig, update, interval=1, blit=True)

fig.tight_layout()
plt.show()
