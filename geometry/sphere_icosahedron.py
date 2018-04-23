import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(9527)

t = 0.5 * (1 + np.sqrt(5))
vertices = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                     [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                     [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]])
vertices /= np.sqrt(1 + t**2)

faces = np.array([[0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [0, 11, 5],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]],
                 dtype=np.uint64)


print('%d nodes, %d faces' % (vertices.shape[0], faces.shape[0]))
num_iter = 3
for ni in range(num_iter):
    faces_new = faces.copy()
    mid_nodes = np.zeros((0, 3), dtype=np.uint64)
    for f in faces:
        m = np.zeros(3, dtype=np.uint64)
        for i in range(3):
            j = ((i + 1) % 3)
            c_ij = (f[i] == mid_nodes[:, 0]) & (f[j] == mid_nodes[:, 1])
            c_ji = (f[j] == mid_nodes[:, 0]) & (f[i] == mid_nodes[:, 1])
            idx = c_ij | c_ji
            if np.any(idx):
                m[i] = mid_nodes[idx, -1]
            else:
                p_mid = np.mean(vertices[[f[i], f[j]], :], axis=0)
                p_mid /= np.linalg.norm(p_mid)

                m[i] = vertices.shape[0]

                vertices = np.vstack((vertices, p_mid[np.newaxis, :]))
                mid_nodes = np.vstack((mid_nodes, [f[i], f[j], m[i]]))

        faces_new[np.all(faces_new == f, axis=1), :] = m
        faces_new = np.vstack((faces_new,
                               [f[0], m[0], m[2]],
                               [f[1], m[1], m[0]],
                               [f[2], m[2], m[1]])).astype(np.uint64)

    faces = faces_new

    print('%d nodes, %d faces' % (vertices.shape[0], faces.shape[0]))


fig = plt.figure(1, figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.axis('scaled')

ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces,
                shade=True,
                color=0.5 * np.ones(3))

axlim = 1.1
ax.set_xlim(-axlim, axlim)
ax.set_ylim(-axlim, axlim)
ax.set_zlim(-axlim, axlim)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('# Iteration = %d' % num_iter)
fig.tight_layout()
plt.show()
