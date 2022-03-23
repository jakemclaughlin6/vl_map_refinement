#pragma once

#include <deque>

#include <TCVL/pose_lookup.h>

#include <beam_calibration/CameraModel.h>
#include <beam_calibration/TfTree.h>
#include <beam_cv/trackers/KLTracker.h>
#include <beam_mapping/Poses.h>
#include <beam_utils/utils.h>

#include <fuse_graphs/hash_graph.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2/buffer_core.h>

#include <fuse_variables/orientation_3d_stamped.h>
#include <fuse_variables/point_3d_landmark.h>
#include <fuse_variables/position_3d_stamped.h>

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
                    const std::string &pose_frame_id,
                    const ros::Time &start_time, const ros::Time &end_time);

  void AddLidarScan(const sensor_msgs::PointCloud2::Ptr &scan);

  void ProcessImage(const cv::Mat &image, const ros::Time &timestamp);

  void OptimizeGraph();

  void OutputResults(const std::string &folder);

  size_t GetNumKeyframes();

protected:
  void AddBaselinkPose(const Eigen::Matrix4d &T_WORLD_BASELINK,
                       const ros::Time &timestamp);

  void AddLandmark(const Eigen::Vector3d &landmark, const uint64_t &id);

  void AddReprojectionConstraint(const ros::Time &pose_time,
                                 const uint64_t &lm_id,
                                 const Eigen::Vector2d &pixel);

  fuse_variables::Position3DStamped::SharedPtr
  GetPosition(const ros::Time &stamp);

  fuse_variables::Orientation3DStamped::SharedPtr
  GetOrientation(const ros::Time &stamp);

  fuse_variables::Point3DLandmark::SharedPtr GetLandmark(uint64_t landmark_id);

  bool GetCameraPose(const ros::Time &stamp, Eigen::Matrix4d &T_WORLD_CAMERA);

  void ProcessLidarCoupling(const ros::Time &kf_time);

  Eigen::Matrix4d PerturbPose(const Eigen::Matrix4d &T_WORLD_SENSOR,
                              const ros::Time &pose_time);

private:
  std::shared_ptr<beam_calibration::TfTree> tree_;
  std::shared_ptr<tcvl::PoseLookup> pose_lookup_;
  std::shared_ptr<beam_calibration::CameraModel> cam_model_;

  std::deque<ros::Time> previous_keyframes_;
  fuse_core::Graph::SharedPtr graph_;
  std::shared_ptr<beam_cv::KLTracker> tracker_;
  std::string camera_frame_id_;
  std::string lidar_frame_id_;
  std::string pose_frame_id_;

  Eigen::Matrix4d T_cam_baselink_;
  Eigen::Matrix4d T_lidar_cam_;
  Eigen::Matrix4d T_baselink_lidar_;

  ros::Time start_time_;
  ros::Time end_time_;

  double total_trajectory_m_;

  size_t num_keyframes_{0};

  std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>
      current_cloud_; // in world frame
};
} // namespace tcvl
