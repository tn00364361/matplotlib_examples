#! /usr/bin/python3
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure


# number of cells in each axis
N = 25
axlim = 1 + 2 / (N - 2)
t = np.linspace(-axlim, axlim, N)
xyz = np.stack(np.meshgrid(t, t, t), axis=0)

f1 = np.linalg.norm(xyz, 2, axis=0)
f2 = np.linalg.norm(xyz, 6, axis=0)
if 1:
    alpha = expit(2 * xyz[-1, ...])
else:
    alpha = (xyz[-1, ...] + 1) / 2
    alpha[alpha > 1] = 1
    alpha[alpha < 0] = 0

f = alpha * f1 + (1 - alpha) * f2

# Use marching cubes to obtain the surface mesh
verts, faces, normals, values = measure.marching_cubes(f, 1.0)
# convert indicies to coordinates
verts = axlim * (2 * verts / (N - 1) - 1)

s = (1 + np.sqrt(5)) / 2
verts[:, 0] *= s
verts[:, 2] /= s

print(verts[:, 0].min(), verts[:, 0].max())

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                triangles=faces,
                shade=True,
                color=0.5 * np.ones(3))

ax.set_xlim(-axlim, axlim)
ax.set_ylim(-axlim, axlim)
ax.set_zlim(-axlim, axlim)
# ax.set_title('p-norm Unit Ball (p = {:.2f})'.format(p))

fig.tight_layout()
plt.show()
