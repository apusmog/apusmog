import torch
import datetime
import math
import time
import sys
import numpy as np
import os
from utils.io import save_checkpoint
from utils.losses import compute_losses
from utils.misc import SmoothedValue

from utils.util import save_to_txt

from third_party.pointnet2 import pointnet2_utils

def compute_learning_rate(args, curr_epoch_normalized):

    assert 1.0 >= curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    elif args.lr_scheduler == 'cosine':
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    elif args.lr_scheduler == 'exp':
        # Exp decay Learning Rate Schedule
        k = 5
        lrate = args.base_lr * math.exp(-k * curr_epoch_normalized)
        curr_lr = args.final_lr + lrate
    else:
        raise ValueError('lr scheduler not implemented')
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch_refinement(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    criterion_upsampling,
    dataset_loader,
    logger
):

    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)

    model.train()

    count = 0.0
    train_loss = 0.0
    losses = {}
    losses_upsampling = {}
    losses_upsampling_ref = {}
    losses_dict = {}
    losses_dict_upsampling = {}
    losses_dict_upsampling_ref = {}

    for item in dataset_loader:
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        data = item
        label = item['label'].to(net_device)

        batch_size = label.size()[0]
        optimizer.zero_grad()

        if len(criterion) > 0:
            logits, parametr, _, _, enc_features = model(data)
            loss, losses_dict = compute_losses(criterion, logits, label, args.lambda_loss)
            loss.backward(retain_graph=True)

        if len(criterion_upsampling) > 0:
            logits_up, sam_parametr, _, _, _, logits_up_ref = model(data, upsampling=args.upsampling_factor)
            loss_upsampling, losses_dict_upsampling = compute_losses(criterion_upsampling, logits_up, data["label_up"].to(net_device),
                                                         args.lambda_loss_upsampling)
            loss_upsampling.backward(retain_graph=True)
            loss_upsampling_ref, losses_dict_upsampling_ref = compute_losses(criterion_upsampling, logits_up_ref, data["label_up"].to(net_device),
                                                         args.lambda_loss_upsampling)
            loss_upsampling_ref.backward()

        optimizer.step()

        count += batch_size
        if len(criterion) > 0:
            train_loss += loss.item() * batch_size
        if len(criterion_upsampling) > 0:
            train_loss += loss_upsampling.item() * batch_size
            train_loss += loss_upsampling_ref.item() * batch_size

        # output losses
        for key in losses_dict:
            if key not in losses:
                losses[key] = 0.0
            losses[key] += losses_dict[key].item() * batch_size

        # upsampling losses
        for key in losses_dict_upsampling:
            if key not in losses_upsampling:
                losses_upsampling[key] = 0.0
            losses_upsampling[key] += losses_dict_upsampling[key].item() * batch_size
        for key in losses_dict_upsampling_ref:
            if key not in losses_upsampling_ref:
                losses_upsampling_ref[key] = 0.0
            losses_upsampling_ref[key] += losses_dict_upsampling_ref[key].item() * batch_size

        time_delta.update(time.time() - curr_time)

        # logging
        if curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

            logger.cprint(
                f"Epoch [{curr_epoch + 1}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Tot loss {train_loss / count:0.5f}; "
                f"LR {curr_lr:0.2e}; "
                f"Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB")

            logger.cprint('Reconstr')
            for loss_key in losses:
                logger.cprint(f"{loss_key} {losses[loss_key] / count:0.6f}")
            logger.cprint('Upsampling')
            for loss_key in losses_upsampling:
                logger.cprint(f"{loss_key} {losses_upsampling[loss_key] / count:0.6f}")
            for loss_key in losses_upsampling_ref:
                logger.cprint(f"{loss_key} {losses_upsampling_ref[loss_key] / count:0.6f}")

            train_loss = 0.0
            count = 0.0
            losses = {}
            losses_upsampling = {}
            losses_upsampling_ref = {}

        curr_iter += 1

    return


@torch.no_grad()
def evaluate_qualit_compose(
    args,
    curr_epoch,
    model,
    dataset_loader,
    curr_train_iter,
    res_dir,
):

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    model.eval()

    epoch_str = f"[{curr_epoch+1}/{args.max_epoch}]" if curr_epoch > 0 else ""

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    torch.set_printoptions(precision=8)

    for idx, data in enumerate(dataset_loader):
        tmp_data = data.copy()
        logits_compose = torch.empty(0)
        logits_up_compose = torch.empty(0)

        curr_time = time.time()

        for p in range(data['patches'].shape[1]):
            tmp_data['pointcloud'] = data['patches'][:, p]
            tmp_data['point_cloud_dims_min'] = data['point_cloud_dims_min'][:, p]
            tmp_data['point_cloud_dims_max'] = data['point_cloud_dims_max'][:, p]

            with torch.no_grad():
                up_factor = args.upsampling_factor
                _, sample_param, _, _, _, logits_up = model(tmp_data, upsampling=up_factor)

            # norm_params = (centroid, max) --> orig data = (pcloud * max) + centroid
            logits_up_inter = logits_up['outputs']['points_logits'] * data['norm_fur_dists'][:, p].cuda() + data['norm_centroids'][:, p].cuda()

            if p == 0:
                logits_up_compose = logits_up_inter
            else:
                logits_up_compose = torch.cat((logits_up_compose, logits_up_inter), 1)

        num_points_rec = data['pointcloud'].shape[1]
        num_points_up = int(data['pointcloud'].shape[1] * args.upsampling_factor)

        inds = pointnet2_utils.furthest_point_sample(logits_up_compose, num_points_up)
        logits_up_compose = logits_up_compose[:, inds[0].long()]
        logits_up_compose = logits_up_compose.squeeze().cpu().numpy()
        time_delta.update(time.time() - curr_time)

        save_to_txt(data['pointcloud'].squeeze().cpu().numpy(), os.path.join(res_dir, '{:04d}_input.xyz'.format(idx)))
        save_to_txt(logits_up_compose,  os.path.join(res_dir, '{:04d}_logits_up.xyz'.format(idx)))

        if curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; Iter time {time_delta.avg:0.6f}; Mem {mem_mb:0.2f}MB"
            )

        curr_iter += 1

    return
