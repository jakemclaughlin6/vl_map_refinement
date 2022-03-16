#include <TCVL/constraints/reprojection_constraint.h>
#include <TCVL/lidar_visual_mapper.h>

#include <beam_cv/detectors/Detectors.h>
#include <beam_cv/geometry/Triangulation.h>
#include <beam_utils/pointclouds.h>
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
  tree_ = std::make_shared<beam_calibration::TfTree>();
  tree_->LoadJSON(extrinsics_path);

  // get pertinent transforms to store
  T_cam_baselink_ =
      tree_->GetTransformEigen(camera_frame_id, pose_frame_id).matrix();
  T_lidar_cam_ =
      tree_->GetTransformEigen(lidar_frame_id, camera_frame_id).matrix();

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
      std::make_shared<beam_cv::FASTDetector>(100);
  tracker_ = std::make_shared<beam_cv::KLTracker>(detector, nullptr, 100);

  // create new graph object
  graph_ = std::make_shared<fuse_graphs::HashGraph>();
}

void LidarVisualMapper::AddLidarScan(sensor_msgs::PointCloud2::Ptr scan) {
  lidar_buffer_.push_back(*scan);
}

void LidarVisualMapper::ProcessImage(const cv::Mat &image,
                                     const ros::Time &timestamp) {
  // add image to tracker
  tracker_->AddImage(image, timestamp);

  // determine if its a keyframe
  if (previous_keyframes_.empty() ||
      timestamp.toSec() - previous_keyframes_.back().toSec() >= 0.25) {
    // push keyframe time to queue
    previous_keyframes_.push_back(timestamp);

    // add keyframes pose to the graph
    Eigen::Matrix4d T_WORLD_BASELINK;
    pose_lookup_->GetT_WORLD_SENSOR(T_WORLD_BASELINK, timestamp);
    AddBaselinkPose(T_WORLD_BASELINK, timestamp);

    size_t num_lms = 0;
    // triangulate possible landmarks and add reproj constraints
    if (previous_keyframes_.size() > 3) {
      std::vector<uint64_t> landmarks =
          tracker_->GetLandmarkIDsInImage(timestamp);
      for (auto &id : landmarks) {
        fuse_variables::Point3DLandmark::SharedPtr lm = GetLandmark(id);
        if (lm) {
          Eigen::Vector2d pixel = tracker_->Get(timestamp, id);
          if (!cam_model_->Undistortable(pixel.cast<int>()))
            continue;
          AddReprojectionConstraint(timestamp, id, pixel);
        } else {

          // get measurements of landmark for triangulation
          std::vector<Eigen::Matrix4d, beam::AlignMat4d> T_cam_world_v;
          std::vector<Eigen::Vector2i, beam::AlignVec2i> pixels;
          std::vector<ros::Time> observation_stamps;
          for (auto &kf_time : previous_keyframes_) {
            try {
              Eigen::Vector2d pixel = tracker_->Get(kf_time, id);
              if (!cam_model_->Undistortable(pixel.cast<int>()))
                continue;
              Eigen::Matrix4d T_CAM_WORLD;
              if (GetCameraPose(kf_time, T_CAM_WORLD)) {
                pixels.push_back(pixel.cast<int>());
                T_cam_world_v.push_back(T_CAM_WORLD.inverse());
                observation_stamps.push_back(kf_time);
              }
            } catch (const std::out_of_range &oor) {
            }
          }

          // triangulate new points
          if (T_cam_world_v.size() >= 3) {
            beam::opt<Eigen::Vector3d> point =
                beam_cv::Triangulation::TriangulatePoint(cam_model_,
                                                         T_cam_world_v, pixels);
            if (point.has_value()) {
              AddLandmark(point.value(), id);
              num_lms++;
              for (int i = 0; i < observation_stamps.size(); i++) {
                AddReprojectionConstraint(
                    observation_stamps[i], id,
                    tracker_->Get(observation_stamps[i], id));
              }
            }
          }
        }
      }
    }

    // manage keyframes
    if (previous_keyframes_.size() > 10) {
      previous_keyframes_.pop_front();
    }
  }

  // TODO:
  // 1. test optimizing visual only graph to make sure it makes sense
  // 2. project lidar map and add V-L constraints (store the 3 best points in
  // the cameras frame)
}

void LidarVisualMapper::OptimizeGraph() {
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = 6;
  options.num_linear_solver_threads = 6;
  options.minimizer_type = ceres::TRUST_REGION;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  graph_->optimize(options);
}

void LidarVisualMapper::AddBaselinkPose(const Eigen::Matrix4d &T_WORLD_BASELINK,
                                        const ros::Time &timestamp) {

  Eigen::Quaterniond q;
  Eigen::Vector3d p;
  beam::TransformMatrixToQuaternionAndTranslation(T_WORLD_BASELINK, q, p);

  fuse_variables::Position3DStamped::SharedPtr position =
      fuse_variables::Position3DStamped::make_shared(timestamp);
  position->x() = p[0];
  position->y() = p[1];
  position->z() = p[2];
  graph_->addVariable(position);

  fuse_variables::Orientation3DStamped::SharedPtr orientation =
      fuse_variables::Orientation3DStamped::make_shared(timestamp);
  orientation->w() = q.w();
  orientation->x() = q.x();
  orientation->y() = q.y();
  orientation->z() = q.z();
  graph_->addVariable(orientation);
}

void LidarVisualMapper::AddLandmark(const Eigen::Vector3d &landmark,
                                    const uint64_t &id) {
  fuse_variables::Point3DLandmark::SharedPtr lm =
      fuse_variables::Point3DLandmark::make_shared(id);
  lm->x() = landmark[0];
  lm->y() = landmark[1];
  lm->z() = landmark[2];
  graph_->addVariable(lm);
}

void LidarVisualMapper::AddReprojectionConstraint(
    const ros::Time &pose_time, const uint64_t &lm_id,
    const Eigen::Vector2d &pixel) {

  fuse_variables::Point3DLandmark::SharedPtr lm = GetLandmark(lm_id);
  fuse_variables::Position3DStamped::SharedPtr position =
      GetPosition(pose_time);
  fuse_variables::Orientation3DStamped::SharedPtr orientation =
      GetOrientation(pose_time);

  if (position && orientation && lm) {
    fuse_constraints::ReprojectionConstraint::SharedPtr vis_constraint =
        fuse_constraints::ReprojectionConstraint::make_shared(
            "TCVL", *orientation, *position, *lm, pixel, T_cam_baselink_,
            cam_model_);
    graph_->addConstraint(vis_constraint);
  }
}

fuse_variables::Position3DStamped::SharedPtr
LidarVisualMapper::GetPosition(const ros::Time &stamp) {
  fuse_variables::Position3DStamped::SharedPtr corr_position =
      fuse_variables::Position3DStamped::make_shared();
  auto corr_position_uuid = fuse_core::uuid::generate(
      corr_position->type(), stamp, fuse_core::uuid::NIL);
  try {
    *corr_position = dynamic_cast<const fuse_variables::Position3DStamped &>(
        graph_->getVariable(corr_position_uuid));
    return corr_position;
  } catch (const std::out_of_range &oor) {
    return nullptr;
  }
}

fuse_variables::Orientation3DStamped::SharedPtr
LidarVisualMapper::GetOrientation(const ros::Time &stamp) {
  fuse_variables::Orientation3DStamped::SharedPtr corr_orientation =
      fuse_variables::Orientation3DStamped::make_shared();
  auto corr_orientation_uuid = fuse_core::uuid::generate(
      corr_orientation->type(), stamp, fuse_core::uuid::NIL);
  try {
    *corr_orientation =
        dynamic_cast<const fuse_variables::Orientation3DStamped &>(
            graph_->getVariable(corr_orientation_uuid));
    return corr_orientation;
  } catch (const std::out_of_range &oor) {
    return nullptr;
  }
}

fuse_variables::Point3DLandmark::SharedPtr
LidarVisualMapper::GetLandmark(uint64_t landmark_id) {
  fuse_variables::Point3DLandmark::SharedPtr landmark =
      fuse_variables::Point3DLandmark::make_shared();
  auto landmark_uuid = fuse_core::uuid::generate(landmark->type(), landmark_id);
  try {
    *landmark = dynamic_cast<const fuse_variables::Point3DLandmark &>(
        graph_->getVariable(landmark_uuid));
    return landmark;
  } catch (const std::out_of_range &oor) {
    return nullptr;
  }
}

bool LidarVisualMapper::GetCameraPose(const ros::Time &stamp,
                                      Eigen::Matrix4d &T_WORLD_CAMERA) {
  fuse_variables::Position3DStamped::SharedPtr p = GetPosition(stamp);
  fuse_variables::Orientation3DStamped::SharedPtr q = GetOrientation(stamp);
  if (p && q) {
    Eigen::Vector3d position(p->data());
    Eigen::Quaterniond orientation(q->w(), q->x(), q->y(), q->z());
    Eigen::Matrix4d T_WORLD_BASELINK;
    beam::QuaternionAndTranslationToTransformMatrix(orientation, position,
                                                    T_WORLD_BASELINK);

    // transform pose from baselink coord space to camera coord space
    T_WORLD_CAMERA = T_WORLD_BASELINK * T_cam_baselink_.inverse();
    return true;
  } else {
    return false;
  }
}

} // namespace tcvl
