import torch
import torch.nn as nn
from utils import torch_tensor_functions


def batch_pairwise_dist(x, y):
  bs, num_points_x, points_dim = x.size()
  _, num_points_y, _ = y.size()
  xx = x.pow(2).sum(dim=-1)
  yy = y.pow(2).sum(dim=-1)
  zz = torch.bmm(x, y.transpose(2, 1))
  rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
  ry = yy.unsqueeze(1).expand_as(zz)
  dists = (rx.transpose(2, 1) + ry - 2 * zz)
  return dists


class AugmChamferLoss(nn.Module):
  def __init__(self):
    super(AugmChamferLoss, self).__init__()
    self.use_cuda = torch.cuda.is_available()

  def forward(self, preds, gts):
    P = batch_pairwise_dist(gts, preds)
    mins, _ = torch.min(P, 1)
    loss_1 = torch.mean(mins)  # torch.sum(mins)
    mins, _ = torch.min(P, 2)
    loss_2 = torch.mean(mins)

    return torch.maximum(loss_1, loss_2)


def loss_on_proj(p0, p1):
  # p0 : B, M, 3
  # p1 : B, N, 3
  thisbatchsize = p0.size()[0]
  output = 0
  dis_map = torch_tensor_functions.compute_sqrdis_map(p0, p1)  # B, M, N

  neighbour_id_01 = torch.topk(dis_map, k=5, dim=-1, largest=False)[1]
  neighbour_dis_01 = torch.topk(dis_map, k=5, dim=-1, largest=False)[0]
  neighbour_id_01 = neighbour_id_01[:, :, 1:]
  neighbour_coor_01 = torch_tensor_functions.indexing_neighbor(p1, neighbour_id_01)
  neighbour_dis_01 = neighbour_dis_01[:, :, 1:]
  neighbour_weight_01 = neighbour_dis_01.detach() * 1000
  neighbour_weight_01 = torch.exp(-1 * neighbour_weight_01)
  neighbour_weight_01 = neighbour_weight_01 / (torch.sum(neighbour_weight_01, dim=-1, keepdim=True) + 0.00001)
  dis_01 = p0.view(thisbatchsize, -1, 1, 3) - neighbour_coor_01
  dis_01 = torch.sum(torch.mul(dis_01, dis_01), dim=-1, keepdim=False)
  pro_dis_01 = torch.mul(neighbour_weight_01, dis_01)
  output += 0.5 * torch.sum(pro_dis_01)

  neighbour_id_10 = torch.topk(dis_map, k=5, dim=1, largest=False)[1].transpose(2, 1)
  neighbour_dis_10 = torch.topk(dis_map, k=5, dim=1, largest=False)[0].transpose(2, 1)
  neighbour_id_10 = neighbour_id_10[:, :, 1:]
  neighbour_coor_10 = torch_tensor_functions.indexing_neighbor(p0, neighbour_id_10)
  neighbour_dis_10 = neighbour_dis_10[:, :, 1:]
  neighbour_weight_10 = neighbour_dis_10.detach() * 1000
  neighbour_weight_10 = torch.exp(-1 * neighbour_weight_10)
  neighbour_weight_10 = neighbour_weight_10 / (torch.sum(neighbour_weight_10, dim=-1, keepdim=True) + 0.00001)
  dis_10 = p1.view(thisbatchsize, -1, 1, 3) - neighbour_coor_10
  dis_10 = torch.sum(torch.mul(dis_10, dis_10), dim=-1, keepdim=False)
  pro_dis_10 = torch.mul(neighbour_weight_10, dis_10)
  output += 0.5 * torch.sum(pro_dis_10)

  return output / thisbatchsize


# Augmented Chamfer (TearingNet)
def augmchamfer(pred, gold):
  loss_cham = AugmChamferLoss()
  pred_coords = pred['outputs']['points_logits'] if type(pred) is dict else pred  # b, n, c
  assert pred_coords.shape[1] == gold.shape[1]

  return loss_cham(pred_coords, gold)


def proj(pred, gold):
  pred_coords = pred['outputs']['points_logits'] if type(pred) is dict else pred  # b, n, c
  loss = loss_on_proj(pred_coords, gold)

  return loss


def compute_losses(loss_list, pred, gold, lambdas=None):

  if loss_list is None or len(loss_list) == 0:
    return None, {}

  tot_loss = 0.0
  loss_dict = {}
  if lambdas is None:
    lambdas = [1] * len(loss_list)

  for i, loss_name in enumerate(loss_list):
    loss = eval(loss_name + '(pred, gold)') * lambdas[i]  # ex: call augmchamfer(pred,gold)
    tot_loss += loss.squeeze()
    loss_dict[loss_name] = loss

  return tot_loss, loss_dict
