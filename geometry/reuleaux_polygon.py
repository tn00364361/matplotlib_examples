import matplotlib.pyplot as plt
import numpy as np


def rot(pt, theta):
    c, s = np.sin(theta), np.cos(theta)
    return np.array([[c, -s], [s, c]]) @ pt

num_edges = 3
alpha = np.deg2rad(360 / num_edges / 4)
theta = np.linspace(alpha, -alpha, 1000, endpoint=False)
r = 2 * np.cos(alpha)
pt = []
for k in range(n):
    pt.append(r * np.array([np.cos(theta), np.sin(theta)]))
    pt[-1][0, :] -= 1

    pt[-1] = rot(pt[-1], np.deg2rad(k * 360 / num_edges))

pt = np.hstack(pt)

fig = plt.figure(1, figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(pt[0, :], pt[1, :])

ax.axis('scaled')
plt.show()
