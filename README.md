# vl_map_refinement

## Description:

This repo implements trajectory refinement using typical visual reprojection constraints and tightly coupled visual-lidar constraints, given an initial trajectory estimate (from odometry or SLAM)

## Dependencies:

* fuse: https://github.com/locusrobotics/fuse (kinetic-devel branch for ROS Kinetic, devel branch for ROS Melodic)
* libbeam: https://github.com/BEAMRobotics/libbeam

## Known Issues:

When compiling a conflict emerges with the rosbag.h file, the solution can be found [here](https://github.com/ethz-asl/lidar_align/issues/16)