import collections

import numpy as np
import pygraph.algorithms
import pygraph.algorithms.minmax
import pygraph.classes.graph
from scipy import spatial


def calc_distances(p0, points):
  return ((p0 - points) ** 2).sum(axis=1)


def furthest_sampling(pts, k):
  farthest_pts = np.zeros((k, 3))
  idxs = [np.random.randint(len(pts))]
  farthest_pts[0] = pts[idxs[0]]
  distances = calc_distances(farthest_pts[0], pts)
  for i in range(1, k):
    farthest_pts[i] = pts[np.argmax(distances)]
    idxs.append(np.argmax(distances))
    distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
  return farthest_pts, np.array(idxs)


class GKNN():
  def __init__(self, data, patch_size=256, patch_num=48):

    self.data = data

    self.clean_data = self.data.copy()

    self.patch_size = patch_size
    self.patch_num = patch_num

    self.nbrs = spatial.cKDTree(self.clean_data)
    dists, idxs = self.nbrs.query(self.clean_data, k=16)

    self.graph2 = pygraph.classes.graph.graph()
    self.graph2.add_nodes(range(len(self.clean_data)))
    sid = 0
    for idx, dist in zip(idxs, dists):
      for eid, d in zip(idx, dist):
        if not self.graph2.has_edge((sid, eid)) and eid < len(self.clean_data):
          self.graph2.add_edge((sid, eid), d)
      sid = sid + 1

    return

  def bfs_knn(self, seed=0, patch_size=1024):
    q = collections.deque()
    visited = set()
    result = []
    q.append(seed)
    while len(visited) < patch_size and q:
      vertex = q.popleft()
      if vertex not in visited:
        visited.add(vertex)
        result.append(vertex)
        if len(q) < patch_size * 5:
          q.extend(self.graph[vertex] - visited)
    return result

  def geodesic_knn(self, seed=0, patch_size=256):
    _, dist = pygraph.algorithms.minmax.shortest_path(self.graph2, seed)

    dist_list = np.asarray([dist[item] if item in dist else 10000 for item in range(len(self.data))])
    idx = np.argsort(dist_list)
    return idx[:patch_size]

  def crop_patch(self):

    _, seeds = furthest_sampling(self.data, self.patch_num)

    patches = []

    i = -1
    for seed in seeds:
      i = i + 1
      patch_size = self.patch_size
      try:
        idx = self.geodesic_knn(seed, patch_size)
      except:
        print("has exception")
        continue
      point = self.data[idx]

      patches.append(point)

    return patches
