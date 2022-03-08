#pragma once

#include <deque>

#include <TCVL/pose_lookup.h>
#include <TCVL/utils.h>

#include <beam_calibration/CameraModel.h>
#include <beam_calibration/TfTree.h>
#include <beam_cv/trackers/KLTracker.h>
#include <beam_mapping/Poses.h>
#include <beam_utils/utils.h>

#include <tf2/buffer_core.h>
#include <fuse_graphs/hash_graph.h>
#include <sensor_msgs/PointCloud2.h>

namespace tcvl {

class LidarVisualMapper {
public:
  /**
   * @brief Constructor
   */
  LidarVisualMapper(const std::string &cam_model_path,
                    const std::string &pose_lookup_path,
                    const std::string &extrinsics_path,
                    const std::string &camera_frame_id,
                    const std::string &lidar_frame_id,
                    const std::string &pose_frame_id);

  void AddLidarScan(sensor_msgs::PointCloud2::ConstPtr scan);

  void ProcessImage(const cv::Mat &image, const ros::Time &timestamp);

private:
  std::shared_ptr<beam_calibration::TfTree> tree_;
  std::shared_ptr<tcvl::PoseLookup> pose_lookup_;
  std::shared_ptr<beam_calibration::CameraModel> cam_model_;

  std::deque<sensor_msgs::PointCloud2> lidar_buffer_;
  fuse_core::Graph::SharedPtr local_graph_;
  std::shared_ptr<beam_cv::KLTracker> tracker_;
  std::string camera_frame_id_;
  std::string lidar_frame_id_;
  std::string pose_frame_id_;
};
} // namespace tcvl
