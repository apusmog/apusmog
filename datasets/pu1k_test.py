import numpy as np
from torch.utils.data import Dataset
import os
from utils import patch_extractor_ecnet


def load_paths(dir):
  paths = []
  assert os.path.isdir(dir), '%s is not a valid directory' % dir

  for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
      if fname.endswith('.xyz') or fname.endswith('.h5'):
        path = os.path.join(root, fname)
        paths.append(path)

  return paths


def pc_normalize(pc, params=None):
  if params is None:
    centroid = np.mean(pc, axis=0)
    pc -= centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  
    pc /= m
  else:
    centroid = params[0]
    pc -= centroid
    m = params[1]
    pc /= m
  return pc, (centroid, m)


def load_data_xyz(fname):
  xyz = open(fname, "r")
  pc_coords = []
  for line in xyz:
    x, y, z = line.split()
    pc_coords.append([float(x), float(y), float(z)])
  xyz.close()

  pc_coords = np.array(pc_coords)

  return pc_coords


class PU1KPatchesTestDataset(Dataset):
  def __init__(self, data_path, args, pt_norm=False):

    inp_dir = os.path.split(data_path)[1]  # ex. input_2048
    data_path_com = os.path.join(data_path, inp_dir)

    self.data_paths = sorted(load_paths(data_path_com))
    self.num_pts_patches = args.num_pts_patches

  def __getitem__(self, item):
    data_path = self.data_paths[item]
    pointcloud = load_data_xyz(data_path)

    gm = patch_extractor_ecnet.GKNN(pointcloud, patch_size=self.num_pts_patches)
    patches = gm.crop_patch()

    norm_patches = []
    norm_centroids = []
    norm_fur_dists = []
    point_cloud_dims_min = []
    point_cloud_dims_max = []
    for patch in patches:
      patch_, params = pc_normalize(patch)
      norm_patches.append(patch_)
      norm_centroids.append(params[0])
      norm_fur_dists.append(params[1])

      point_cloud_dims_min.append(patch_.min(axis=0))
      point_cloud_dims_max.append(patch_.max(axis=0))

    ret_dict = {}
    ret_dict["pointcloud"] = pointcloud.astype(np.float32)
    ret_dict["patches"] = np.array(norm_patches).astype(np.float32)
    ret_dict["norm_centroids"] = np.array(norm_centroids).astype(np.float32)
    ret_dict["norm_fur_dists"] = np.array(norm_fur_dists).astype(np.float32)
    ret_dict["pointcloud_path"] = data_path
    ret_dict["point_cloud_dims_min"] = np.array(point_cloud_dims_min).astype(np.float32)
    ret_dict["point_cloud_dims_max"] = np.array(point_cloud_dims_max).astype(np.float32)

    return ret_dict

  def __len__(self):
    return len(self.data_paths)
