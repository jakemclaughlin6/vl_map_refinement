#include <TCVL/lidar_visual_mapper.h>

//#include <beam_cv/geometry/Triangulation.h>
#include <beam_cv/detectors/Detectors.h>
#include <boost/filesystem.hpp>

namespace tcvl {

LidarVisualMapper::LidarVisualMapper(const std::string &cam_model_path,
                        const std::string &pose_lookup_path,
                        const std::string &extrinsics_path,
                        const std::string &camera_frame_id,
                        const std::string &lidar_frame_id,
                        const std::string &pose_frame_id)
    : camera_frame_id_(camera_frame_id), lidar_frame_id_(lidar_frame_id),
      pose_frame_id_(pose_frame_id) {

  // Load ground truth poses file
  if (!boost::filesystem::exists(pose_lookup_path) ||
      boost::filesystem::is_directory(pose_lookup_path)) {
    BEAM_ERROR("Invalid Poses File Path.");
    throw std::runtime_error{"Invalid Poses File Path."};
  }
  pose_lookup_ = std::make_shared<tcvl::PoseLookup>(
      tcvl::LoadPoses(pose_lookup_path), pose_frame_id, "world");

  // Load robot extrinsics calibrations
  if (!boost::filesystem::exists(extrinsics_path) ||
      boost::filesystem::is_directory(extrinsics_path)) {
    BEAM_ERROR("Invalid Extrinsics File Path.");
    throw std::runtime_error{"Invalid Extrinsics File Path."};
  }
  tree_->LoadJSON(extrinsics_path);

  // Load main camera intrinsics
  if (!boost::filesystem::exists(cam_model_path) ||
      boost::filesystem::is_directory(cam_model_path)) {
    BEAM_ERROR("Invalid Camera Intrinsics File Path.");
    throw std::runtime_error{"Invalid Camera Intrinsics File Path."};
  }
  std::string cam_model_path_copy = cam_model_path;
  cam_model_ = beam_calibration::CameraModel::Create(cam_model_path_copy);

  // Initialize visual tracker
  std::shared_ptr<beam_cv::Detector> detector =
      std::make_shared<beam_cv::FASTDetector>();
  tracker_ = std::make_shared<beam_cv::KLTracker>(detector, nullptr, 100);
}

void LidarVisualMapper::AddLidarScan(sensor_msgs::PointCloud2::ConstPtr scan) {}

void LidarVisualMapper::ProcessImage(const cv::Mat &image, const ros::Time &timestamp) {}

} // namespace tcvl
