#!/usr/bin/bash

#PRED_DIR="$1"
#GT_XYZ_DIR="$2"
#GT_OFF_DIR="$3"

# Create Conda environment and install Tensorflow

conda create -n pu1k_eval python=3.9
conda activate pu1k_eval
pip install tensorflow

# Build the code for computing point-to-mesh distance
# This requires CGAL (sudo apt-get install libcgal-dev)

if [ -d "./build" ]; then
	cd ./build
else
	mkdir ./build
	cd ./build
	cmake ..
fi
make
cd ..

# Compute point-to-mesh distance for each sample

#python compute_p2m.py --pred_dir $PRED_DIR --gt_dir $GT_OFF_DIR

# Compute metrics

#python evaluate_tf_cpu.py --pred_dir $PRED_DIR --gt_dir $GT_XYZ_DIR --save_path results