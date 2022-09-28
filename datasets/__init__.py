from .pu1k_patches import PU1KPatchesDataset
from .pu1k_test import PU1KPatchesTestDataset

from torch.utils.data import DataLoader
import os

DATASET_FUNCTIONS = {
  "pu1k": [PU1KPatchesDataset],
  "pu1k_test": [PU1KPatchesTestDataset],
}


def build_dataloaders(args, partition):
  dataset = DATASET_FUNCTIONS[args.dataset_name][0]

  data_dir = os.path.join(args.dataset_root_dir, args.dataset_dir)

  data_path_train = os.path.join(data_dir, 'train')
  data_path_test = os.path.join(data_dir, 'test', args.test_data_dir)

  train = DataLoader(dataset(data_path=data_path_train, args=args),
                     num_workers=args.dataset_num_workers, batch_size=args.batchsize_per_gpu, shuffle=True,
                     drop_last=True) if partition == 'train' else []
  test = DataLoader(dataset(data_path=data_path_test, args=args),
                    num_workers=args.dataset_num_workers, batch_size=args.batchsize_per_gpu, shuffle=False,
                    drop_last=False) if partition == 'test' else []

  dataloader_dict = {
    "train": train,
    "test": test
  }
  return dataloader_dict
