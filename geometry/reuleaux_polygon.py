#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as geo


num_edges = 7

assert np.mod(num_edges, 2) == 1

r = 2 * np.cos(np.pi / num_edges / 2)
polygon = geo.box(-1, -1, 1, 1)
for theta in np.linspace(0, 2 * np.pi, num_edges, endpoint=False):
    circle = geo.Point(np.cos(theta), np.sin(theta)).buffer(r, resolution=1024)
    polygon = polygon.intersection(circle)

pt = np.asarray(polygon.exterior.coords.xy)

fig = plt.figure(1, figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

ax.plot(pt[0, :], pt[1, :], '.')

ax.axis('scaled')
plt.show()
