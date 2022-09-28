import torch
import torch.nn as nn
from models.helpers import GenericMLP

from third_party.lib_pointtransformer.pointops.functions import pointops


class PointTransformerLayer(nn.Module):
  def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
    super().__init__()
    self.mid_planes = mid_planes = out_planes // 1
    self.out_planes = out_planes
    self.share_planes = share_planes
    self.nsample = nsample
    self.linear_q = nn.Linear(in_planes, mid_planes)
    self.linear_k = nn.Linear(in_planes, mid_planes)
    self.linear_v = nn.Linear(in_planes, out_planes)
    self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
    self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                  nn.Linear(mid_planes, mid_planes // share_planes),
                                  nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                  nn.Linear(out_planes // share_planes, out_planes // share_planes))
    self.softmax = nn.Softmax(dim=1)

  def forward(self, pxo) -> torch.Tensor:

    p, x, o = pxo  # (n, 3), (n, c), (b)
    x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
    x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
    x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
    p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
    for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1,
                                                                                                      2).contiguous() if i == 1 else layer(
      p_r)  # (n, nsample, c)
    w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes,
                                          self.mid_planes).sum(2)  # (n, nsample, c)
    for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1,
                                                                                                  2).contiguous() if i % 3 == 0 else layer(
      w)
    w = self.softmax(w)  # (n, nsample, c)
    n, nsample, c = x_v.shape
    s = self.share_planes
    x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
    return x


class PointTransformerBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, share_planes=8, nsample=16):
    super(PointTransformerBlock, self).__init__()
    self.transformer2 = PointTransformerLayer(in_planes, planes, share_planes, nsample)
    self.bn2 = nn.BatchNorm1d(planes)
    self.relu = nn.ReLU(inplace=True)


  def forward(self, pxo):
    p, x, o = pxo  # (n, 3), (n, c), (b)

    x = self.relu(self.bn2(self.transformer2([p, x, o])))

    return [p, x.squeeze(), o]


class PointTransfFE(nn.Module):
  def __init__(self, block=PointTransformerBlock, blocks=1, c=3, out_planes=128, share_planes=8, nsample=16):
    super().__init__()
    self.c = c
    self.in_planes, planes = self.c, out_planes
    self.enc1 = self._make_enc(block, planes, blocks, share_planes, nsample=nsample)

  def _make_enc(self, block, planes, blocks, share_planes=8, nsample=16):
    layers = []
    out_planes = planes * block.expansion
    for _ in range(blocks):
      layers.append(block(self.in_planes, out_planes, share_planes, nsample=nsample))
    return nn.Sequential(*layers)

  def forward(self, pxo):
    bsize = pxo.shape[0]
    npts = pxo.shape[1]
    p0 = pxo.view(-1, self.c).clone()
    x0 = pxo.view(-1, self.c).clone()
    o0 = torch.IntTensor([npts*(i+1) for i in range(bsize)]).cuda()
    p1, x1, o1 = self.enc1([p0, x0, o0])

    out = x1.view(bsize, npts, -1).permute(0, 2, 1)

    return pxo, out, None
