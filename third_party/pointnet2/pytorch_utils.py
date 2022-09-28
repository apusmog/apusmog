
''' Modified based on Ref: https://github.com/erikwijmans/Pointnet2_PyTorch '''
import torch
import torch.nn as nn
from typing import List, Tuple


class SharedMLP(nn.Sequential):

  def __init__(
      self,
      args: List[int],
      *,
      bn: bool = False,
      activation=nn.ReLU(inplace=True),
      preact: bool = False,
      first: bool = False,
      name: str = "",
      remove_last_activation=False
  ):
    super().__init__()

    last_act = len(args) - 2 if remove_last_activation else -1
    for i in range(len(args) - 1):
      self.add_module(
        name + 'layer{}'.format(i),
        Conv2d(
          args[i],
          args[i + 1],
          bn=(not first or not preact or (i != 0)) and bn,
          activation=activation if (not first or not preact or i != 0) and i != last_act else None,  # and i != last_act
          preact=preact
        )
      )


class _BNBase(nn.Sequential):

  def __init__(self, in_size, batch_norm=None, name=""):
    super().__init__()
    self.add_module(name + "bn", batch_norm(in_size))

    nn.init.constant_(self[0].weight, 1.0)
    nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

  def __init__(self, in_size: int, *, name: str = ""):
    super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

  def __init__(self, in_size: int, name: str = ""):
    super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class BatchNorm3d(_BNBase):

  def __init__(self, in_size: int, name: str = ""):
    super().__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class _ConvBase(nn.Sequential):

  def __init__(
      self,
      in_size,
      out_size,
      kernel_size,
      stride,
      padding,
      activation,
      bn,
      init,
      conv=None,
      batch_norm=None,
      bias=True,
      preact=False,
      name=""
  ):
    super().__init__()

    bias = bias and (not bn)
    conv_unit = conv(
      in_size,
      out_size,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      bias=bias
    )
    init(conv_unit.weight)
    if bias:
      nn.init.constant_(conv_unit.bias, 0)

    if bn:
      if not preact:
        bn_unit = batch_norm(out_size)
      else:
        bn_unit = batch_norm(in_size)

    if preact:
      if bn:
        self.add_module(name + 'bn', bn_unit)

      if activation is not None:
        self.add_module(name + 'activation', activation)

    self.add_module(name + 'conv', conv_unit)

    if not preact:
      if bn:
        self.add_module(name + 'bn', bn_unit)

      if activation is not None:
        self.add_module(name + 'activation', activation)


class Conv1d(_ConvBase):

  def __init__(
      self,
      in_size: int,
      out_size: int,
      *,
      kernel_size: int = 1,
      stride: int = 1,
      padding: int = 0,
      activation=nn.ReLU(inplace=True),
      bn: bool = False,
      init=nn.init.kaiming_normal_,
      bias: bool = True,
      preact: bool = False,
      name: str = ""
  ):
    super().__init__(
      in_size,
      out_size,
      kernel_size,
      stride,
      padding,
      activation,
      bn,
      init,
      conv=nn.Conv1d,
      batch_norm=BatchNorm1d,
      bias=bias,
      preact=preact,
      name=name
    )


class Conv2d(_ConvBase):

  def __init__(
      self,
      in_size: int,
      out_size: int,
      *,
      kernel_size: Tuple[int, int] = (1, 1),
      stride: Tuple[int, int] = (1, 1),
      padding: Tuple[int, int] = (0, 0),
      activation=nn.ReLU(inplace=True),
      bn: bool = False,
      init=nn.init.kaiming_normal_,
      bias: bool = True,
      preact: bool = False,
      name: str = ""
  ):
    super().__init__(
      in_size,
      out_size,
      kernel_size,
      stride,
      padding,
      activation,
      bn,
      init,
      conv=nn.Conv2d,
      batch_norm=BatchNorm2d,
      bias=bias,
      preact=preact,
      name=name
    )


class Conv3d(_ConvBase):

  def __init__(
      self,
      in_size: int,
      out_size: int,
      *,
      kernel_size: Tuple[int, int, int] = (1, 1, 1),
      stride: Tuple[int, int, int] = (1, 1, 1),
      padding: Tuple[int, int, int] = (0, 0, 0),
      activation=nn.ReLU(inplace=True),
      bn: bool = False,
      init=nn.init.kaiming_normal_,
      bias: bool = True,
      preact: bool = False,
      name: str = ""
  ):
    super().__init__(
      in_size,
      out_size,
      kernel_size,
      stride,
      padding,
      activation,
      bn,
      init,
      conv=nn.Conv3d,
      batch_norm=BatchNorm3d,
      bias=bias,
      preact=preact,
      name=name
    )


class FC(nn.Sequential):

  def __init__(
      self,
      in_size: int,
      out_size: int,
      *,
      activation=nn.ReLU(inplace=True),
      bn: bool = False,
      init=None,
      preact: bool = False,
      name: str = ""
  ):
    super().__init__()

    fc = nn.Linear(in_size, out_size, bias=not bn)
    if init is not None:
      init(fc.weight)
    if not bn:
      nn.init.constant_(fc.bias, 0)

    if preact:
      if bn:
        self.add_module(name + 'bn', BatchNorm1d(in_size))

      if activation is not None:
        self.add_module(name + 'activation', activation)

    self.add_module(name + 'fc', fc)

    if not preact:
      if bn:
        self.add_module(name + 'bn', BatchNorm1d(out_size))

      if activation is not None:
        self.add_module(name + 'activation', activation)


def set_bn_momentum_default(bn_momentum):
  def fn(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
      m.momentum = bn_momentum

  return fn


class BNMomentumScheduler(object):

  def __init__(
      self, model, bn_lambda, last_epoch=-1,
      setter=set_bn_momentum_default
  ):
    if not isinstance(model, nn.Module):
      raise RuntimeError(
        "Class '{}' is not a PyTorch nn Module".format(
          type(model).__name__
        )
      )

    self.model = model
    self.setter = setter
    self.lmbd = bn_lambda

    self.step(last_epoch + 1)
    self.last_epoch = last_epoch

  def step(self, epoch=None):
    if epoch is None:
      epoch = self.last_epoch + 1

    self.last_epoch = epoch
    self.model.apply(self.setter(self.lmbd(epoch)))

  def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.shape[0]
    n = xyz1.shape[1]
    c = xyz1.shape[2]
    m = xyz2.shape[1]
    xyz1 = torch.tile(torch.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    xyz2 = torch.tile(torch.reshape(xyz2, (b,m,1,c)), [1,1,n,1])


    # xyz1 = torch.expand(xyz1, dim=1)
    # xyz2 = tf.expand_dims(xyz2, axis=2)
    dist = torch.sum((xyz1 - xyz2) ** 2, -1)
    # dist = tf.sqrt(tf.abs(dist))
    # outi, out = select_top_k(k, dist)
    # idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    # val = tf.slice(out, [0,0,0], [-1,-1,k])

    val, idx = tf.nn.top_k(-dist, k=k)  # ONLY SUPPORT CPU
    return val,
