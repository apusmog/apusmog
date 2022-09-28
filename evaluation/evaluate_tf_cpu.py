import argparse
import os
import numpy as np
import tensorflow as tf
import glob
import csv
from collections import OrderedDict
import math
import time

tf.compat.v1.disable_eager_execution()

# Utility functions

def load(filename):
    return np.loadtxt(filename).astype(np.float32)

def normalize_point_cloud(pc):
    
    centroid = tf.reduce_mean(pc, axis = 1, keepdims = True)
    pc = pc - centroid
    furthest_distance = tf.reduce_max(
        tf.sqrt(tf.reduce_sum(pc ** 2, axis = -1, keepdims = True)), axis = 1, keepdims = True)
    pc = pc / furthest_distance
    
    return pc, centroid, furthest_distance

def nn_distance_cpu(pc1, pc2):
    
    N, M = pc1.get_shape()[1], pc2.get_shape()[1]
    pc1_expand_tile = tf.tile(tf.expand_dims(pc1, 2), [1, 1, M, 1])
    pc2_expand_tile = tf.tile(tf.expand_dims(pc2, 1), [1, N, 1, 1])
    pc_diff = pc1_expand_tile - pc2_expand_tile # [B, N, M, C]
    pc_dist = tf.reduce_sum(pc_diff ** 2, axis = -1) # [B, N, M]
    dist1 = tf.reduce_min(pc_dist, axis = 2) # [B, N]
    idx1 = tf.argmin(pc_dist, axis = 2) # [B, N]
    dist2 = tf.reduce_min(pc_dist, axis = 1) # [B, M]
    idx2 = tf.argmin(pc_dist, axis = 1) # [B, M]
    
    return dist1, idx1, dist2, idx2

# Read predicted and GT point clouds

parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir", type = str, required = True, help = ".xyz")
parser.add_argument("--gt_dir", type = str, required = True, help = ".xyz")
parser.add_argument("--use_p2f", action='store_true')
parser.add_argument("--save_path", type = str, required = True, help = "Path to save the result")
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok = True)

print('Loading predicted point clouds from:', args.pred_dir)
pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, '*_logits_up.xyz')))
#pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, '*.xyz')))
p2f_paths = []
if args.use_p2f:
    print(args.use_p2f)
    p2f_paths = sorted(glob.glob(os.path.join(args.pred_dir, '*_point2mesh_distance.xyz')))
    pred_paths = [item for item in pred_paths if item not in p2f_paths]  
    print(len(p2f_paths), len(pred_paths))
    assert len(p2f_paths) == len(pred_paths)

print('Loading GT point clouds from:', args.gt_dir)
gt_paths = sorted(glob.glob(os.path.join(args.gt_dir, '*.xyz')))
assert len(pred_paths) == len(gt_paths)

print('Found %d point clouds' % len(pred_paths))

gt_pred_pairs = [(gt, pred) for gt, pred in zip(gt_paths, pred_paths)

gt = load(gt_paths[0])[:, :3]
pred_placeholder = tf.compat.v1.placeholder(tf.float32, [1, gt.shape[0], 3])
gt_placeholder = tf.compat.v1.placeholder(tf.float32, [1, gt.shape[0], 3])

pred_tensor, centroid, furthest_distance = normalize_point_cloud(pred_placeholder)
gt_tensor, centroid, furthest_distance = normalize_point_cloud(gt_placeholder)

cd_forward, _, cd_backward, _ = nn_distance_cpu(pred_tensor, gt_tensor)
cd_forward = cd_forward[0, :]
cd_backward = cd_backward[0, :]

start = time.time()

with tf.compat.v1.Session() as sess:
    fieldnames = ["name", "CD", "hausdorff", "hausdorff_new", "p2f avg", "p2f std"]

    avg_md_forward_value = 0
    avg_md_backward_value = 0
    avg_hd_value = 0
    avg_hd_value_new = 0
    counter = 0

    global_p2f = []

    with open(os.path.join(args.save_path, "evaluation.csv"), "w") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
        writer.writeheader()

        # Generate per-sample metrics

        for i, (gt_path, pred_path) in enumerate(gt_pred_pairs):

            print('Evaluating %d / %d point cloud...' % (i + 1, len(gt_paths)))

            print(pred_path)
            print(gt_path)

            pred_name = os.path.splitext(os.path.basename(pred_path))[0]
            gt_name = os.path.splitext(os.path.basename(gt_path))[0]
            #assert gt_name in pred_name

            row = {}
            gt = load(gt_path)[:, :3]
            gt = gt[np.newaxis, ...]
            pred = load(pred_path)
            pred = pred[:, :3]

            row["name"] = os.path.basename(gt_path)
            pred = pred[np.newaxis, ...]
            cd_forward_value, cd_backward_value = sess.run([cd_forward, cd_backward],
                                                           feed_dict={pred_placeholder: pred, gt_placeholder: gt})

            # Chamfer and Hausdorff distances

            # print(cd_forward_value.shape)
            # print(cd_backward_value.shape)

            md_value = np.mean(cd_forward_value) + np.mean(cd_backward_value)
            hd_value = np.max(np.amax(cd_forward_value, axis=0) + np.amax(cd_backward_value, axis=0))
            hd_value_new = max(np.max(cd_forward_value), np.max(cd_backward_value))
            cd_backward_value = np.mean(cd_backward_value)
            cd_forward_value = np.mean(cd_forward_value)
            row["CD"] = cd_forward_value + cd_backward_value
            row["hausdorff"] = hd_value
            row["hausdorff_new"] = hd_value_new
            avg_md_forward_value += cd_forward_value
            avg_md_backward_value += cd_backward_value
            avg_hd_value += hd_value
            avg_hd_value_new += hd_value_new

            # Point-to-mesh distance

            if p2f_paths:
                point2mesh_distance = load(p2f_paths[i])
                if point2mesh_distance.size != 0:
                    point2mesh_distance = point2mesh_distance[:, 3]
                    row["p2f avg"] = np.nanmean(point2mesh_distance)
                    row["p2f std"] = np.nanstd(point2mesh_distance)
                    global_p2f.append(point2mesh_distance)

            writer.writerow(row)
            counter += 1

        # Generate final metrics

        row = OrderedDict()
        avg_md_forward_value /= counter
        avg_md_backward_value /= counter
        avg_hd_value /= counter
        avg_hd_value_new /= counter
        avg_cd_value = avg_md_forward_value + avg_md_backward_value
        row["CD"] = avg_cd_value
        row["hausdorff"] = avg_hd_value
        row["hausdorff_new"] = avg_hd_value_new

        if global_p2f:
            global_p2fs = np.concatenate(global_p2f, axis=0)
            mean_p2f = np.nanmean(global_p2fs)
            std_p2f = np.nanstd(global_p2fs)
            row["p2f avg"] = mean_p2f
            row["p2f std"] = std_p2f

        writer.writerow(row)

        row = OrderedDict()
        row["CD (1e-3)"] = avg_cd_value * 1000.
        row["hausdorff (1e-3)"] = avg_hd_value * 1000.
        row["hausdorff_new (1e-3)"] = avg_hd_value_new * 1000.

        if global_p2f:
            global_p2fs = np.concatenate(global_p2f, axis=0)
            mean_p2f = np.nanmean(global_p2fs)
            std_p2f = np.nanstd(global_p2fs)
            row["p2f avg (1e-3)"] = mean_p2f * 1000.
            row["p2f std (1e-3)"] = std_p2f * 1000.

        print("{:60s} ".format("name"), "|".join(["{:>15s}".format(d) for d in fieldnames[1:]]))
        print("{:60s}".format(os.path.abspath(args.pred_dir)), end=' ')
        print(" | ".join(["{:>15.8f}".format(d) for d in row.values()]))

        with open(os.path.join(args.save_path, "finalresult.text"), "w") as text:
            print(row, file=text)

end = time.time()
print('Total execution time: {} s'.format(end - start))
