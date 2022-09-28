import h5py
import numpy as np
from torch.utils.data import Dataset
import random
import os


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


def translate_pointcloud(pointcloud, trans_vec):
  translated_pointcloud = np.add(pointcloud, trans_vec).astype('float32')

  return translated_pointcloud


def scale_pointcloud(pointcloud, scale):
  scaled_pointcloud = np.multiply(pointcloud, scale).astype('float32')

  return scaled_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.0025, clip=0.02):
  sig = random.uniform(0.0, 0.01)
  N, C = pointcloud.shape
  pointcloud += np.clip(sig * np.random.randn(N, C), -1 * clip, clip)

  return pointcloud


def Rmatrix(gamma, beta, alpha):
  return np.array([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)],
                   [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma),  np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
                   [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]]).astype(np.float32)


def rotate_pointcloud(pointcloud, angles=None):
  if angles is None:
    roll = np.random.uniform(0, 2 * np.pi)
    pitch = np.random.uniform(0, 2 * np.pi)
    yaw = np.random.uniform(0, 2 * np.pi)
    angles = (roll, pitch, yaw)

  rot_matrix = Rmatrix(angles[0], angles[1], angles[2])
  rot_pointcloud = np.matmul(rot_matrix, pointcloud.transpose()).transpose()

  return rot_pointcloud


def load_data_h5(fname, num_points):
  with h5py.File(fname, "r") as f:
    a_group_key = list(f.keys())[0] if num_points == 1024 else list(f.keys())[1]
    data = f[a_group_key][:]  # list(f[a_group_key])

  return data


class PU1KPatchesDataset(Dataset):
  def __init__(self, data_path, args):

    self.num_points = args.pointcloud_npoints

    self.partition = 'train'
    self.data_paths = sorted(load_paths(data_path))
    self.data_h5_input = load_data_h5(self.data_paths[0], args.num_pts_patches)  # only one h5 file in train
    self.data_h5_gt = load_data_h5(self.data_paths[0], int(args.num_pts_patches * args.upsampling_factor))

    print(self.partition, data_path)

    self.noise_aug = args.noise_aug
    self.rotation_aug = args.rotation_aug
    self.scale_aug = args.scale_aug
    self.trans_aug = args.trans_aug

    self.up_factor = args.upsampling_factor
    self.eps = 1e-8

  def __getitem__(self, item):
    pointcloud = self.data_h5_input[item]
    label_up = self.data_h5_gt[item]
    data_path = self.data_paths[0]

    label_up, norm_params = pc_normalize(label_up)
    pointcloud, _ = pc_normalize(pointcloud, norm_params)
    label = pointcloud.copy()

    if random.uniform(0, 1) < self.rotation_aug:   # random rotation
      roll = np.random.uniform(0, 2 * np.pi)
      pitch = np.random.uniform(0, 2 * np.pi)
      yaw = np.random.uniform(0, 2 * np.pi)
      pointcloud = rotate_pointcloud(pointcloud, (roll, pitch, yaw))
      label = pointcloud.copy()
      label_up = rotate_pointcloud(label_up, (roll, pitch, yaw))
    else:
      pointcloud = rotate_pointcloud(pointcloud, (self.eps, self.eps, self.eps))

    if random.uniform(0, 1) < self.noise_aug:
      pointcloud = jitter_pointcloud(pointcloud)  # random jittering
    if random.uniform(0, 1) < self.scale_aug:
      scale = np.random.uniform(low=0.8, high=1.2, size=[3])
      pointcloud = scale_pointcloud(pointcloud, scale)  # random scaling
      label = pointcloud.copy()
      label_up = scale_pointcloud(label_up, scale)
    if random.uniform(0, 1) < self.trans_aug:
      trans_vec = np.random.uniform(low=-0.05, high=0.05, size=[3])
      pointcloud = translate_pointcloud(pointcloud, trans_vec)  # random scaling
      label = pointcloud.copy()
      label_up = translate_pointcloud(label_up, trans_vec)

    # random shuffling
    np.random.shuffle(pointcloud)

    point_cloud_dims_min = pointcloud.min(axis=0)
    point_cloud_dims_max = pointcloud.max(axis=0)

    ret_dict = {}
    ret_dict["pointcloud_id"] = item
    ret_dict["pointcloud"] = pointcloud.astype(np.float32)
    ret_dict["label"] = label.astype(np.float32)
    ret_dict["label_up"] = label_up.astype(np.float32)
    ret_dict["pointcloud_path"] = data_path
    ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
    ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)

    return ret_dict

  def __len__(self):
    return len(self.data_h5_input)

