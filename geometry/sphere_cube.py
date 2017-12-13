import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


vtxs = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]], dtype=np.float64)
vtxs /= np.sqrt(3)
squares = np.array([[0, 4, 5, 1], [7, 4, 0, 3], [5, 4, 7, 6],
                [1, 2, 3, 0], [6, 2, 1, 5], [3, 2, 6, 7]])

print('%d nodes, %d squares' % (vtxs.shape[0], squares.shape[0]))
mid_nodes = np.zeros((0, 3), dtype=np.uint64)
nIter = 3
for ni in range(nIter):
    squares_new = []
    for s in squares:
        m = np.zeros(5, dtype=np.uint64)
        for i in range(4):
            j = (i + 1) % 4
            c_ij = (s[i] == mid_nodes[:, 0]) & (s[j] == mid_nodes[:, 1])
            c_ji = (s[j] == mid_nodes[:, 0]) & (s[i] == mid_nodes[:, 1])
            if np.any(c_ij | c_ji):
                m[i] = mid_nodes[c_ij | c_ji, -1]
            else:
                p_mid = vtxs[s[i], :] + vtxs[s[j], :]
                p_mid /= np.linalg.norm(p_mid)
                vtxs = np.vstack((vtxs, p_mid[np.newaxis, :]))
                m[i] = vtxs.shape[0] - 1
                mid_nodes = np.vstack((mid_nodes, [s[i], s[j], m[i]]))
        p_center = np.mean(vtxs[s, :], axis=0)
        p_center /= np.linalg.norm(p_center)
        vtxs = np.vstack((vtxs, p_center[np.newaxis, :]))
        m[-1] = vtxs.shape[0] - 1

        squares_new.append([
            [s[0], m[0], m[-1], m[3]], [s[1], m[1], m[-1], m[0]],
            [s[2], m[2], m[-1], m[1]], [s[3], m[3], m[-1], m[2]]
        ])

    squares = np.vstack(squares_new).astype(np.uint64)

    print('%d nodes, %d squares' % (vtxs.shape[0], squares.shape[0]))

faces = []
for s in squares:
    faces.append([[s[0], s[1], s[2]], [s[0], s[2], s[3]]])

faces = np.vstack(faces)

fig = plt.figure(1, figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.axis('scaled')

ax.plot_trisurf(vtxs[:, 0], vtxs[:, 1], vtxs[:, 2],
    triangles=faces, shade=True, color=0.5 * np.ones(3))
axlim = 1.1
ax.set_xlim(-axlim, axlim)
ax.set_ylim(-axlim, axlim)
ax.set_zlim(-axlim, axlim)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('# Iteration = %d' % nIter)
fig.tight_layout()
plt.show()

#savefig(str(nIter) + '.png', dpi=135)
