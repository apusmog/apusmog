import argparse
import os
import sys

import numpy as np
import torch
from torch.multiprocessing import set_start_method
import random

from datasets import build_dataloaders
from engine import evaluate_qualit_compose, train_one_epoch_refinement
from models import build_model
from optimizer import build_optimizer
from utils.io import save_checkpoint, resume_if_possible
from utils.cfg_utils import load_cfg_from_cfg_file, merge_cfg_from_list
from utils.util import IOStream


def make_args_parser():
  parser = argparse.ArgumentParser("Point Cloud Upsampling Using Transformers", add_help=False)

  parser.add_argument('--config', type=str, required=True, help='config file')
  parser.add_argument('opts', help='see config/*.yaml for all options', default=None, nargs=argparse.REMAINDER)
  argums = parser.parse_args()
  assert argums.config is not None
  config = load_cfg_from_cfg_file(argums.config)
  if argums.opts is not None:
    config = merge_cfg_from_list(config, argums.opts)

  return config


def do_train(
    args,
    model,
    optimizer,
    criterion,
    criterion_upsampling,
    dataloaders,
):
  """
    Main training loop.
    This trains the model for `args.max_epoch` epochs.
    """

  num_iters_per_epoch = len(dataloaders["train"])
  # print(f"Model is {model}")
  print(f"Training started at epoch {args.start_epoch} until {args.max_epoch}.")
  print(f"One training epoch = {num_iters_per_epoch} iters.")

  logger = IOStream(os.path.join("./checkpoints", args.checkpoint_dir, 'logs.txt'))
  logger.cprint(f"{args}")
  logger.cprint('Tot params: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

  for epoch in range(args.start_epoch, args.max_epoch):
    train_one_epoch_refinement(
      args,
      epoch,
      model,
      optimizer,
      criterion,
      criterion_upsampling,
      dataloaders["train"],
      logger
    )

    # latest checkpoint is always stored in checkpoint.pth
    save_checkpoint(
      os.path.join("./checkpoints", args.checkpoint_dir),
      model,
      optimizer,
      epoch,
      args,
      filename="checkpoint.pth",
    )

    curr_iter = epoch * len(dataloaders["train"])
    print("==" * 10)
    print(f"Epoch [{epoch+1}/{args.max_epoch}];")
    print("==" * 10)

    if (
        epoch > 0
        and args.save_separate_checkpoint_every_epoch > 0
        and epoch % args.save_separate_checkpoint_every_epoch == 0
    ):
      # separate checkpoints are stored as checkpoint_{epoch}.pth
      save_checkpoint(
        os.path.join("./checkpoints", args.checkpoint_dir),
        model,
        optimizer,
        epoch,
        args,
      )


def test_model_qualit_compose(args, model, dataloaders):

  if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
    print(f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}")
    sys.exit(1)

  sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
  model.load_state_dict(sd["model"])
  epoch = -1
  curr_iter = 0
  evaluate_qualit_compose(args, epoch, model, dataloaders["test"], curr_iter,
                       "./checkpoints/" + args.checkpoint_dir + '/results')


def main(local_rank, args):
  # print(f"Called with args: {args}")

  torch.cuda.set_device(local_rank)
  np.random.seed(args.seed)
  random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

  phase = 'test' if args.test_qualit_compose else 'train'
  dataloaders = build_dataloaders(args, phase)
  model = build_model(args)
  model = model.cuda(local_rank)
  print('Tot params: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

  criterion = args.loss   # list of losses
  criterion_upsampling = args.loss_upsampling

  if args.test_qualit_compose:
    test_model_qualit_compose(args, model, dataloaders)
  else:   # train model
    assert (
        args.checkpoint_dir is not None
    ), f"Please specify a checkpoint dir using --checkpoint_dir"
    if not os.path.isdir(args.checkpoint_dir):
      os.makedirs("./checkpoints/" + args.checkpoint_dir, exist_ok=True)
    optimizer = build_optimizer(args, model)
    loaded_epoch = resume_if_possible(
      "./checkpoints/" + args.checkpoint_dir, model, optimizer
    )
    args.start_epoch = loaded_epoch + 1
    do_train(
      args,
      model,
      optimizer,
      criterion,
      criterion_upsampling,
      dataloaders,
    )


if __name__ == "__main__":
  cfg = make_args_parser()
  try:
    set_start_method("spawn")
  except RuntimeError:
    pass

  main(local_rank=0, args=cfg)
