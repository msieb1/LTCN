#!/bin/bash

# Source environment
. env.sh

# Training TCN for single views
#python run.py -m tcn_no_captions -e red_cube_stacking -r single_view -b SingleViewTripletBuilder

# Training PoseNetwork for single views of multiple time steps
python run.py -m pose -e duck_oracle -r single_view -b SingleViewPoseBuilder
#python run.py -m pose -e duck_pose_freebody -r single_view -b SingleViewPoseBuilder
#python run.py -m pose_euler_crop -e duck_pose -r single_view -b SingleViewPoseBuilder

# Training ViewNetwork for multiple views for one time step
#python run.py -m view -e rubix_multiview -r multi_view -b OneViewQuaternionBuilder
#python run.py -m view -e duck_multiview_camera -r multi_view -b OneViewQuaternionBuilder


