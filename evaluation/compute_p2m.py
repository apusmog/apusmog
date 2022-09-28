import argparse
import os
import numpy as np
import glob
import math
import time
import subprocess
import multiprocessing as mp

def work(pair):
    pred, gt = pair
    cmd = ['./build/evaluation', gt, pred]
    subprocess.run(cmd, capture_output = True)

parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir", type = str, required = True)
parser.add_argument("--gt_dir", type = str, required = True)
parser.add_argument("--use_mp", type = bool, default = False)
args = parser.parse_args()

print('Loading predicted point clouds from:', args.pred_dir)
pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, '*_logits_up.xyz')))
print('Loading GT point clouds from:', args.gt_dir)
gt_paths = sorted(glob.glob(os.path.join(args.gt_dir, '*.off')))

assert len(pred_paths) == len(gt_paths)
print('Found %d point clouds' % len(pred_paths))

start = time.time()

if args.use_mp:
    pool = mp.Pool(processes = mp.cpu_count())
    pool.map(work, zip(pred_paths, gt_paths))
else:
    logfile = open('p2m_log.txt', 'w')
    for pred, gt in zip(pred_paths, gt_paths):
        print('Computing point-to-mesh distance for:', pred)
        p2m_args = ['./build/evaluation', gt, pred]
        output = subprocess.check_output(p2m_args, universal_newlines = True)
        logfile.write(output)

end = time.time()
print('Total execution time: {} s'.format(end - start))
