import torch
import torch.nn
import json
import numpy as np
from PIL import Image
from third_party.pointnet2 import pointnet2_utils

pi = torch.acos(torch.zeros(1)).item() * 2
eps = 1e-6


def cart2sph(x, y, z):
  hxy = np.hypot(x, y)
  r = np.hypot(hxy, z)
  el = np.arctan2(z, hxy)
  az = np.arctan2(y, x)
  return az, el, r


def sph2cart(az, el, r):
  rcos_theta = r * np.cos(el)
  x = rcos_theta * np.cos(az)
  y = rcos_theta * np.sin(az)
  z = r * np.sin(el)
  return x, y, z

#
# def calc_distances(p0, points):
#   return ((p0 - points) ** 2).sum(axis=1)
#
#
# def calc_distances_harvesine(p0, points):
#   p0 = p0[np.newaxis]
#   return haversine_distances(points, p0)


# def furthest_sampling(pts, K):
#   farthest_pts = np.zeros((K, pts.shape[1]))
#   farthest_pts[0] = pts[np.random.randint(len(pts))]
#   distances = calc_distances(farthest_pts[0], pts)
#   for i in range(1, K):
#     farthest_pts[i] = pts[np.argmax(distances)]
#     distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
#   return farthest_pts


# def furthest_sampling_harvesine(pts, K):
#   farthest_pts = np.zeros((K, pts.shape[1]))
#   farthest_pts[0] = pts[np.random.randint(len(pts))]
#   distances = calc_distances_harvesine(farthest_pts[0], pts)
#   for i in range(1, K):
#     farthest_pts[i] = pts[np.argmax(distances)]
#     distances = np.minimum(distances, calc_distances_harvesine(farthest_pts[i], pts))
#   return farthest_pts
#
#
# def estimate_density_and_sample(points, num_sampling):
#   kde = KernelDensity(bandwidth=0.035, kernel='gaussian').fit(points) #metric='haversine'
#
#   # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
#   #                     {'bandwidth': np.linspace(0.001, 0.5, 30)},
#   #                     cv=20)
#   # grid.fit(points)
#   # print(grid.best_params_)
#   # kde = grid.best_estimator_
#
#   sampled_points = kde.sample(num_sampling)
#   # sampled_points[:, 1] = np.clip(sampled_points[:, 1], -1, 1)
#   # sampled_points[:, 1] = np.arcsin(sampled_points[:, 1])
#
#   return sampled_points


def knn(x, k):
  inner = -2 * torch.matmul(x.transpose(2, 1), x)
  xx = torch.sum(x ** 2, dim=1, keepdim=True)
  pairwise_distance = -xx - inner - xx.transpose(2, 1)

  idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
  return idx


def save_to_txt(tosave, fname):
  if torch.is_tensor(tosave):
    tosave = tosave.squeeze().cpu().numpy()

  np.savetxt(fname, tosave, delimiter=" ")   #.xyz


def write_json(data_to_write, filename):
  with open(filename, 'w') as f:
    json.dump(data_to_write, f, indent=4)


def write_res_file(filename, res_dict):
  print(filename)
  data_to_write = 'Test metrics\n'
  for loss_key in res_dict:
    data_to_write += '{} = {:.8f}\n'.format(loss_key, res_dict[loss_key])
  with open(filename, 'a') as f:
    f.write(data_to_write)


# create a file and write the text into it
class IOStream():
  def __init__(self, path):
    self.f = open(path, 'a')

  def cprint(self, text):
    print(text)
    self.f.write(text + '\n')
    self.f.flush()

  def close(self):
    self.f.close()


# -----------------------------------------------------------------------------
# Functions for parsing args
# -----------------------------------------------------------------------------
import yaml
import os
from ast import literal_eval
import copy


class CfgNode(dict):
  """
  CfgNode represents an internal node in the configuration tree. It's a simple
  dict-like container that allows for attribute-based access to keys.
  """

  def __init__(self, init_dict=None, key_list=None, new_allowed=False):
    # Recursively convert nested dictionaries in init_dict into CfgNodes
    init_dict = {} if init_dict is None else init_dict
    key_list = [] if key_list is None else key_list
    for k, v in init_dict.items():
      if type(v) is dict:
        # Convert dict to CfgNode
        init_dict[k] = CfgNode(v, key_list=key_list + [k])
    super(CfgNode, self).__init__(init_dict)

  def __getattr__(self, name):
    if name in self:
      return self[name]
    else:
      raise AttributeError(name)

  def __setattr__(self, name, value):
    self[name] = value

  def __str__(self):
    def _indent(s_, num_spaces):
      s = s_.split("\n")
      if len(s) == 1:
        return s_
      first = s.pop(0)
      s = [(num_spaces * " ") + line for line in s]
      s = "\n".join(s)
      s = first + "\n" + s
      return s

    r = ""
    s = []
    for k, v in sorted(self.items()):
      seperator = "\n" if isinstance(v, CfgNode) else " "
      attr_str = "{}:{}{}".format(str(k), seperator, str(v))
      attr_str = _indent(attr_str, 2)
      s.append(attr_str)
    r += "\n".join(s)
    return r

  def __repr__(self):
    return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def load_cfg_from_cfg_file(file):
  cfg = {}
  assert os.path.isfile(file) and file.endswith('.yaml'), \
    '{} is not a yaml file'.format(file)

  with open(file, 'r') as f:
    cfg_from_file = yaml.safe_load(f)

  for key in cfg_from_file:
    for k, v in cfg_from_file[key].items():
      cfg[k] = v

  cfg = CfgNode(cfg)
  return cfg


def merge_cfg_from_list(cfg, cfg_list):
  new_cfg = copy.deepcopy(cfg)
  assert len(cfg_list) % 2 == 0
  for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
    subkey = full_key.split('.')[-1]
    assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
    value = _decode_cfg_value(v)
    value = _check_and_coerce_cfg_value_type(
      value, cfg[subkey], subkey, full_key
    )
    setattr(new_cfg, subkey, value)

  return new_cfg


def _decode_cfg_value(v):
  """Decodes a raw config value (e.g., from a yaml config files or command
  line argument) into a Python object.
  """
  # All remaining processing is only applied to strings
  if not isinstance(v, str):
    return v
  # Try to interpret `v` as a:
  #   string, number, tuple, list, dict, boolean, or None
  try:
    v = literal_eval(v)
  # The following two excepts allow v to pass through when it represents a
  # string.
  #
  # Longer explanation:
  # The type of v is always a string (before calling literal_eval), but
  # sometimes it *represents* a string and other times a data structure, like
  # a list. In the case that v represents a string, what we got back from the
  # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
  # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
  # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
  # will raise a SyntaxError.
  except ValueError:
    pass
  except SyntaxError:
    pass
  return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
  """Checks that `replacement`, which is intended to replace `original` is of
  the right type. The type is correct if it matches exactly or is one of a few
  cases in which the type can be easily coerced.
  """
  original_type = type(original)
  replacement_type = type(replacement)

  # The types must match (with some exceptions)
  if replacement_type == original_type:
    return replacement

  # Cast replacement from from_type to to_type if the replacement and original
  # types match from_type and to_type
  def conditional_cast(from_type, to_type):
    if replacement_type == from_type and original_type == to_type:
      return True, to_type(replacement)
    else:
      return False, None

  # Conditionally casts
  # list <-> tuple
  casts = [(tuple, list), (list, tuple)]
  # For py2: allow converting from str (bytes) to a unicode string
  try:
    casts.append((str, unicode))  # noqa: F821
  except Exception:
    pass

  for (from_type, to_type) in casts:
    converted, converted_value = conditional_cast(from_type, to_type)
    if converted:
      return converted_value

  raise ValueError(
    "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
    "key: {}".format(
      original_type, replacement_type, original, replacement, full_key
    )
  )


def _assert_with_logging(cond, msg):
  if not cond:
    logger.debug(msg)
  assert cond, msg


def find_free_port():
  import socket
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  # Binding to port 0 will cause the OS to find an available port for us
  sock.bind(("", 0))
  port = sock.getsockname()[1]
  sock.close()
  # NOTE: there is still a chance the port could be taken by other processes.
  return port


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):  # n is the batch size， update all variables
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def convert_spherical(input):
  r = torch.norm(input, dim=-1, keepdim=True)
  result = r.clone()
  for i in range(input.shape[-1] - 2):
    result = torch.cat((result, torch.acos(torch.clamp(input[..., i].unsqueeze(-1) / (r+eps), -1.0, 1.0))), -1)
    r = torch.sqrt(torch.square(r) - torch.square(input[..., i]).unsqueeze(-1) + eps)

  acos = torch.acos(torch.clamp(input[..., -2].unsqueeze(-1) / (r+eps), -1.0, 1.0))
  last_angle = torch.where((input[..., -1] >= 0).unsqueeze(-1), acos, 2*pi - acos)
  result = torch.cat((result, last_angle), -1)

  return result


def convert_rectangular(input):
  r = input[..., 0].unsqueeze(-1)
  multi_sin = torch.ones_like(r)
  for i in range(1, input.shape[-1] - 1):
    x_i = r * multi_sin * torch.cos(input[..., i]).unsqueeze(-1)
    result = x_i if i == 1 else torch.cat((result, x_i), dim=-1)
    multi_sin = multi_sin * torch.sin(input[..., i]).unsqueeze(-1)
  result = torch.cat((result, r * multi_sin * torch.cos(input[..., -1]).unsqueeze(-1)), dim=-1)
  result = torch.cat((result, r * multi_sin * torch.sin(input[..., -1]).unsqueeze(-1)), dim=-1)

  return result
