# TCVL

## Description:

This repo implements a case study on different types of tightly coupled visual to lidar constraints.
This repository makes use of the [M2DGR](https://github.com/SJTU-ViSYS/M2DGR) dataset with calibrations converted into libbeam's formatting.

## Dependencies:

* fuse: https://github.com/locusrobotics/fuse (kinetic-devel branch for ROS Kinetic, devel branch for ROS Melodic)
* libbeam: https://github.com/BEAMRobotics/libbeam

## Known Issues:

When compiling a conflict emerges with the rosbag.h file, the solution can be found [here](https://github.com/ethz-asl/lidar_align/issues/16)