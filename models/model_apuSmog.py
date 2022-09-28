import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from .pointTransf_featureExtr import PointTransfFE
from .pointTransf_refinement import PointTransfRef

from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)
from utils.util import convert_spherical, convert_rectangular
from torch.distributions import MultivariateNormal
from utils.losses import batch_pairwise_dist


class ModelApuSmog(nn.Module):

  def __init__(
      self,
      pre_encoder,
      encoder,
      decoder,
      refinement,
      args
  ):
    super().__init__()
    self.pre_encoder = pre_encoder
    self.preenc_to_repr = GenericMLP(
      input_dim=args.enc_dim,
      hidden_dims=[args.enc_dim // 2],
      output_dim=3 + 3,  # 3D coords + variances (1 elev, 1 azim, 1 cov)
      norm_fn_name="bn1d",
      activation="relu",
      use_conv=True,
      output_use_activation=False,
      output_use_norm=False,
      output_use_bias=False,
    )
    self.encoder = encoder
    if hasattr(self.encoder, "masking_radius"):
      hidden_dims = [args.enc_dim]
    else:
      hidden_dims = [args.enc_dim, args.enc_dim]
    self.encoder_to_decoder_projection = GenericMLP(
      input_dim=args.enc_dim,
      hidden_dims=hidden_dims,
      output_dim=args.dec_dim,
      norm_fn_name="bn1d",
      activation="relu",
      use_conv=True,
      output_use_activation=True,
      output_use_norm=True,
      output_use_bias=False,
    )
    self.pos_embedding = PositionEmbeddingCoordsSine(
      d_pos=args.dec_dim, pos_type='fourier', normalize=True
    )

    self.query_projection = GenericMLP(
      input_dim=args.enc_dim,
      hidden_dims=[args.dec_dim],
      output_dim=args.dec_dim,
      use_conv=True,
      output_use_activation=True,
      hidden_use_bias=True,
    )
    self.decoder = decoder
    self.build_mlp_heads(args.dec_dim)

    self.num_pts_sphere = args.preenc_npoints
    self.refinement = refinement

  def build_mlp_heads(self, decoder_dim, mlp_dropout=0.0):
    mlp_func = partial(
      GenericMLP,
      norm_fn_name="bn1d",
      activation="relu",
      use_conv=True,
      hidden_dims=[decoder_dim // 2, decoder_dim // 4],
      dropout=mlp_dropout,
      input_dim=decoder_dim,
    )

    out_dim = 3
    upsampling_pc_head = mlp_func(output_dim=out_dim)

    mlp_heads = [
      ("upsampling_pc_head", upsampling_pc_head),
    ]
    self.mlp_heads = nn.ModuleDict(mlp_heads)

  def get_query_embeddings(self, parametr_xyz, point_cloud_dims):
    pos_embed = self.pos_embedding(parametr_xyz.contiguous(), input_range=point_cloud_dims)
    query_embed = self.query_projection(pos_embed)

    return query_embed

  def _break_up_pc(self, pc):
    # pc may contain color/normals.

    xyz = pc[..., 0:3].contiguous()
    features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
    return xyz, features

  def run_encoder(self, point_clouds, up_factor=1):
    xyz, _ = self._break_up_pc(point_clouds)
    pre_enc_xyz, pre_enc_features, _ = self.pre_encoder(xyz)
    # xyz: batch x npoints x 3
    # features: batch x channel x npoints
    # inds: batch x npoints
    nptssphere = self.num_pts_sphere
    bsize = pre_enc_features.shape[0]

    inter_repr = self.preenc_to_repr(pre_enc_features).permute(0, 2, 1)

    coords = normalize(inter_repr[..., :3], dim=-1)  # 3D coords on unit sphere
    logvar_cov = inter_repr[..., 3:]

    idxs = np.empty(0)
    eps = 1e-6
    min_std = 0.05
    max_std = 0.50
    if up_factor > 1:
      mean = convert_spherical(coords)[..., 1:]  # spherical coords 2D
      stds = logvar_cov[..., :2].mul(0.5).exp_() - (1 - (max_std + min_std) / 2)  # mean std closer to 0 than 1
      std_phi = torch.clamp(stds[..., 0], min=min_std)  # clip standard deviations
      std_theta = torch.clamp(stds[..., 1], min=min_std)  # clip standard deviations
      covariance = torch.clamp(logvar_cov[..., -1], min=-(std_phi * std_theta) + eps,
                               max=std_phi * std_theta - eps)  # covariance constraint, see proof in supp

      cov_matr = torch.zeros(bsize, nptssphere, 2, 2).cuda()
      cov_matr[:, :, 0, 0] = std_phi ** 2
      cov_matr[:, :, 1, 1] = std_theta ** 2
      cov_matr[:, :, 0, 1] = cov_matr[:, :, 1, 0] = covariance

      num_samples = int(xyz.shape[1] * up_factor)

      # Sample N*R points randomly
      idxs = sorted((np.random.uniform(0, 1, num_samples) * nptssphere).astype(int).tolist())

      m_ = mean[:, idxs]
      cov_ = cov_matr[:, idxs]
      m = m_.view(-1, 2)
      cov = cov_.view(-1, 2, 2)

      distr = MultivariateNormal(m, cov)
      sample = distr.rsample()
      sample_from_param_ = sample.view(bsize, -1, 2)

      # CLIP DRAWN SAMPLES IN [-ksig, ksig]
      ksig = 2
      sample_from_param = torch.zeros_like(sample_from_param_).cuda()
      sample_from_param[..., 0] = torch.clamp(sample_from_param_[..., 0],
                                              min=m_[..., 0] - ksig * torch.sqrt(cov_[:, :, 0, 0]),
                                              max=m_[..., 0] + ksig * torch.sqrt(cov_[:, :, 0, 0]))
      sample_from_param[..., 1] = torch.clamp(sample_from_param_[..., 1],
                                              min=m_[..., 1] - ksig * torch.sqrt(cov_[:, :, 1, 1]),
                                              max=m_[..., 1] + ksig * torch.sqrt(cov_[:, :, 1, 1]))

      hypersph_sampled = torch.ones(coords.shape[0], num_samples,
                                    coords.shape[2]).cuda()
      hypersph_sampled[..., 1:] = sample_from_param
      sampled_repr = convert_rectangular(hypersph_sampled)
    else:
      sampled_repr = coords

    pre_enc_features = pre_enc_features.permute(2, 0, 1)

    # xyz points are in batch x npointx channel order
    enc_xyz, enc_features, enc_inds = self.encoder(
      pre_enc_features, xyz=pre_enc_xyz
    )

    return enc_xyz, enc_features, enc_inds, sampled_repr, pre_enc_features, idxs

  def get_pts_predictions(self, point_features):

    # point_features change to (num_layers x batch) x channel x num_queries
    point_features = point_features.permute(0, 2, 3, 1)
    num_layers, batch, channel, num_queries = (
      point_features.shape[0],
      point_features.shape[1],
      point_features.shape[2],
      point_features.shape[3],
    )
    point_features = point_features.reshape(num_layers * batch, channel, num_queries)
    # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
    pts_logits = self.mlp_heads["upsampling_pc_head"](point_features).transpose(1, 2)

    # reshape outputs to num_layers x batch x nqueries x noutput
    pts_logits = pts_logits.reshape(num_layers, batch, num_queries, -1)

    outputs = []
    for l in range(num_layers):
      points_prediction = {
        "points_logits": pts_logits[l, :, :, :3]
      }
      outputs.append(points_prediction)

    # intermediate decoder layer outputs are only used during training
    aux_outputs = outputs[:-1]
    outputs = outputs[-1]

    return {
      "outputs": outputs,  # output from last layer of decoder
      "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
    }

  def compute_ref_idxs(self, inputs, preds_up):
    point_clouds = inputs["pointcloud"].cuda()
    dists = batch_pairwise_dist(preds_up, point_clouds)
    idxs = torch.argmin(dists, dim=2).tolist()  # B x NSAMPLES

    return idxs

  def forward(self, inputs, encoder_only=False, upsampling=1):
    point_clouds = inputs["pointcloud"].cuda()

    enc_xyz, enc_features, enc_inds, parametr, preenc_features, _ = self.run_encoder(point_clouds, upsampling)

    enc_features = self.encoder_to_decoder_projection(
      enc_features.permute(1, 2, 0)
    ).permute(2, 0, 1)
    # encoder features: npoints x batch x channel
    # encoder xyz: npoints x batch x 3

    if encoder_only:
      # return: batch x npoints x channels
      return enc_xyz, enc_features.transpose(0, 1)

    point_cloud_dims = [
      inputs["point_cloud_dims_min"].cuda(),
      inputs["point_cloud_dims_max"].cuda(),
    ]

    query_embed = self.get_query_embeddings(parametr, point_cloud_dims)
    # query_embed: batch x channel x npoint
    enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

    # decoder expects: npoints x batch x channel
    enc_pos = enc_pos.permute(2, 0, 1)
    query_embed = query_embed.permute(2, 0, 1)

    tgt = torch.zeros_like(query_embed)
    point_features = self.decoder(tgt, enc_features, query_pos=query_embed, pos=enc_pos)[
      0]
    pts_predictions = self.get_pts_predictions(point_features)
    pts_predictions_ref = {'outputs': {'points_logits': pts_predictions['outputs']['points_logits'].clone()}}
    idxs = self.compute_ref_idxs(inputs, pts_predictions['outputs']['points_logits'])

    if upsampling > 1:
      preds_not_ref = pts_predictions_ref['outputs']['points_logits'].clone()
      pts_predictions_ref['outputs']['points_logits'] = self.refinement(preds_not_ref,
                                                                          preenc_features.detach().permute(1, 2, 0),
                                                                          idxs).permute(0, 2, 1)

      return pts_predictions, parametr, enc_xyz, enc_inds, enc_features, pts_predictions_ref
    else:
      return pts_predictions, parametr, enc_xyz, enc_inds, enc_features


def build_preencoder(args):
  if args.preencoder == 'pointTransformerFeatureExtr':
    preencoder = PointTransfFE(out_planes=args.enc_dim, nsample=args.preenc_nsample)
    return preencoder
  else:
    raise NameError('Preencoder not defined')


def build_encoder(args):

  encoder_layer = TransformerEncoderLayer(
    d_model=args.enc_dim,
    nhead=args.enc_nhead,
    dim_feedforward=args.enc_ffn_dim,
    dropout=args.enc_dropout,
    activation=args.enc_activation,
  )
  encoder = TransformerEncoder(
    encoder_layer=encoder_layer, num_layers=args.enc_nlayers
  )
  return encoder


def build_decoder(args):

  decoder_layer = TransformerDecoderLayer(
    d_model=args.dec_dim,
    nhead=args.dec_nhead,
    dim_feedforward=args.dec_ffn_dim,
    dropout=args.dec_dropout,
  )
  decoder = TransformerDecoder(
    decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
  )
  return decoder


def build_refinement(args):
  if args.refinement == 'pointTransformerRefinement':
    refinement = PointTransfRef(blocks=args.ref_blocks, out_planes=128,
                                              share_planes=32,
                                              nsample=int(args.upsampling_factor * 4))
    return refinement
  else:
    raise NameError('Preencoder not defined')


def build_apuSmog(args):
  pre_encoder = build_preencoder(args)
  encoder = build_encoder(args)
  decoder = build_decoder(args)
  refinement = build_refinement(args)
  model = ModelApuSmog(
    pre_encoder,
    encoder,
    decoder,
    refinement,
    args=args
  )

  return model
