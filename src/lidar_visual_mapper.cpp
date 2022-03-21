#include <TCVL/constraints/reprojection_constraint.h>
#include <TCVL/lidar_visual_mapper.h>

#include <beam_cv/detectors/Detectors.h>
#include <beam_cv/geometry/Triangulation.h>
#include <beam_depth/DepthMap.h>
#include <beam_utils/pointclouds.h>
#include <boost/filesystem.hpp>

namespace tcvl {

LidarVisualMapper::LidarVisualMapper(
    const std::string &cam_model_path, const std::string &pose_lookup_path,
    const std::string &extrinsics_path, const std::string &camera_frame_id,
    const std::string &lidar_frame_id, const std::string &pose_frame_id,
    const ros::Time &start_time, const ros::Time &end_time)
    : camera_frame_id_(camera_frame_id), lidar_frame_id_(lidar_frame_id),
      pose_frame_id_(pose_frame_id), start_time_(start_time),
      end_time_(end_time) {

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
  T_baselink_lidar_ = T_cam_baselink_.inverse() * T_lidar_cam_.inverse();

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

  // initialize local point cloud
  current_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

  // get an average trajectory length for the trajectory
  total_trajectory_m_ = 0;
  int time_length = end_time_.toSec() - start_time_.toSec();
  Eigen::Vector3d previous_position;
  Eigen::Vector3d current_position;
  for (size_t i = 0; i < time_length; i++) {
    ros::Time new_time(start_time_.toSec() + (double)i);
    Eigen::Matrix4d T_WORLD_BASELINK;
    if (!pose_lookup_->GetT_WORLD_SENSOR(T_WORLD_BASELINK, new_time))
      continue;
    Eigen::Vector3d p;
    Eigen::Quaterniond q;
    beam::TransformMatrixToQuaternionAndTranslation(T_WORLD_BASELINK, q, p);
    if (i == 0) {
      previous_position = p;
      current_position = p;
      continue;
    } else {
      previous_position = current_position;
      current_position = p;
    }
    total_trajectory_m_ += beam::distance(current_position, previous_position);
  }
}

void LidarVisualMapper::AddLidarScan(
    const sensor_msgs::PointCloud2::Ptr &scan) {
  // transform into world frame to keep a running local point cloud
  pcl::PointCloud<pcl::PointXYZ> cloud = beam::ROSToPCL(*scan);

  // get point clouds pose
  Eigen::Matrix4d T_WORLD_BASELINK;
  pose_lookup_->GetT_WORLD_SENSOR(T_WORLD_BASELINK, scan->header.stamp);

  // perturb pose and put it into lidars frame
  // PerturbPose(T_WORLD_BASELINK, msg.header.stamp);
  Eigen::Matrix4d T_WORLD_LIDAR = T_WORLD_BASELINK * T_baselink_lidar_;

  // transform scan into world frame
  pcl::PointCloud<pcl::PointXYZ> cloud_in_world_frame;
  pcl::transformPointCloud(cloud, cloud_in_world_frame, T_WORLD_LIDAR);

  // add to aggregate cloud
  *current_cloud_ += cloud_in_world_frame;
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
    num_keyframes_++;

    // add keyframes pose to the graph
    Eigen::Matrix4d T_WORLD_BASELINK;
    pose_lookup_->GetT_WORLD_SENSOR(T_WORLD_BASELINK, timestamp);

    // perturb pose
    // PerturbPose(T_WORLD_BASELINK, timestamp);

    // add pose to graph
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
          Eigen::Vector2i tmp;
          if (!cam_model_->UndistortPixel(pixel.cast<int>(), tmp))
            continue;
          AddReprojectionConstraint(timestamp, id, pixel);
        } else {
          // get measurements of landmark for triangulation
          std::vector<Eigen::Matrix4d, beam::AlignMat4d> T_cam_world_v;
          std::vector<Eigen::Vector2i, beam::AlignVec2i> pixels;
          std::vector<ros::Time> observation_stamps;

          for (auto &kf_time : previous_keyframes_) {
            try {
              Eigen::Vector2i pixel = tracker_->Get(kf_time, id).cast<int>();
              Eigen::Vector2i tmp;
              if (!cam_model_->UndistortPixel(pixel, tmp))
                continue;
              Eigen::Matrix4d T_CAM_WORLD;
              if (GetCameraPose(kf_time, T_CAM_WORLD)) {
                pixels.push_back(pixel);
                T_cam_world_v.push_back(T_CAM_WORLD.inverse());
                observation_stamps.push_back(kf_time);
              }
            } catch (const std::out_of_range &oor) {
            }
          }
          // triangulate new points
          if (T_cam_world_v.size() >= 3) {
            beam::opt<Eigen::Vector3d> point =
                beam_cv::Triangulation::TriangulatePoint(
                    cam_model_, T_cam_world_v, pixels, 5.0, 30);
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

    // add tightly coupled VL constraints
    ProcessLidarCoupling(timestamp);

    // manage keyframes
    if (previous_keyframes_.size() > 20) {
      previous_keyframes_.pop_front();
    }
  }
}

void LidarVisualMapper::ProcessLidarCoupling(const ros::Time &kf_time) {
  int R = 9;
  // // transform into current camera frame
  // Eigen::Matrix4d T_CAM_WORLD;
  // if (GetCameraPose(kf_time, T_CAM_WORLD)) {
  //   // transform the current point cloud into the camera frame
  //   std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud_in_camera_frame =
  //       std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  //   pcl::transformPointCloud(*current_cloud_, *cloud_in_camera_frame,
  //                            T_CAM_WORLD);

  //   // extract depth map using projection
  //   beam_depth::DepthMap dm(cam_model_, cloud_in_camera_frame);
  //   dm.ExtractDepthMapProjection(30);
  //   cv::Mat depth_image = dm.GetDepthImage();

  //   // loop through all landmarks in image and do neighbourhood search
  //   std::vector<uint64_t> landmarks =
  //   tracker_->GetLandmarkIDsInImage(kf_time); for (auto &id : landmarks) {
  //     Eigen::Vector2i pixel = tracker_->Get(kf_time, id).cast<int>();

  //     // compute 4 corners of search area
  //     int start_col = pixel[0] - R, end_col = pixel[0] + R,
  //         start_row = pixel[1] - R, end_row = pixel[1] + R;

  //     // get points in landmark neighbourhood <distance,depth,pixel>
  //     std::vector<std::tuple<double, double, Eigen::Vector2i>>
  //         neighbourhood_points;
  //     for (int col = start_col; col < end_col; col++) {
  //       for (int row = start_row; row < end_row; row++) {
  //         Eigen::Vector2i search_point(col, row);
  //         double depth = depth_image.at<float>(row, col);
  //         if (col < 0 || row < 0 || col > depth_image.cols ||
  //             row > depth_image.rows || depth == 0)
  //           continue;
  //         double distance = beam::distance(pixel, search_point);
  //        // neighbourhood_points.push_back(std::make_tuple(distance, depth));
  //       }
  //     }

  //     // sort by pixel distance to current landmark
  //     std::sort(neighbourhood_points.begin(), neighbourhood_points.end());

  //     // compute mean, median and stddev of the depths
  //     std::vector<double> depths;
  //     for (auto &tuple : neighbourhood_points) {
  //       std::cout << std::get<0>(tuple) << std::endl;
  //       //depths.push_back(d);
  //     }
  //     double median = depths[depths.size() / 2];
  //     double sum = std::accumulate(depths.begin(), depths.end(), 0.0);
  //     double mean = sum / depths.size();
  //     double sq_sum =
  //         std::inner_product(depths.begin(), depths.end(), depths.begin(),
  //         0.0);
  //     double stdev = std::sqrt(sq_sum / depths.size() - mean * mean);

  //     // filter out points that are outside of 1 stddev
  //     std::vector<std::tuple<double, double, Eigen::Vector2i>>
  //         valid_neighbourhood_points;
  //     for (auto &tuple : neighbourhood_points) {
  //       if (std::get<1>(tuple) < median + stdev ||
  //           std::get<1>(tuple) > median - stdev) {
  //         valid_neighbourhood_points.push_back(tuple);
  //       }
  //     }

  //     if (valid_neighbourhood_points.size() >= 3) {
  //       // get 3 closest points
  //       std::vector<Eigen::Vector3d> matching_points;
  //       for (int i = 0; i < 3; i++) {
  //         Eigen::Vector2i pixel = std::get<2>(valid_neighbourhood_points[i]);
  //         double depth = std::get<1>(valid_neighbourhood_points[i]);
  //         Eigen::Vector3d direction;
  //         if (cam_model_->BackProject(pixel, direction)) {
  //           Eigen::Vector3d coords = depth * direction;
  //           matching_points.push_back(coords);
  //         }
  //       }

  //       // TODO: add vl constraint
  //     }
  //   }
  // }
  current_cloud_->points.clear();
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

void LidarVisualMapper::OutputResults(const std::string &folder) {
  if (!boost::filesystem::is_directory(folder)) {
    BEAM_ERROR("Invalid output folder.");
    throw std::runtime_error{"Invalid output folder."};
  }
}

size_t LidarVisualMapper::GetNumKeyframes() { return num_keyframes_; }

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

Eigen::Matrix4d
LidarVisualMapper::PerturbPose(const Eigen::Matrix4d &T_WORLD_SENSOR,
                               const ros::Time &pose_time) {

  Eigen::Vector3d p;
  Eigen::Quaterniond q;
  beam::TransformMatrixToQuaternionAndTranslation(T_WORLD_SENSOR, q, p);
  // perturb position
  double w = (pose_time.toSec() - start_time_.toSec()) /
             (end_time_.toSec() - start_time_.toSec());
  double x = total_trajectory_m_ / 300.0;
  Eigen::Vector3d perturbation(2 * w * x, w * x, w * x);
  p += perturbation;
  // perturb yaw
  double y = total_trajectory_m_ / 100.0;
  auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
  euler[2] += w * y;
  q = Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ());
  // update pose
  Eigen::Matrix4d output;
  beam::QuaternionAndTranslationToTransformMatrix(q, p, output);
  return output;
}
} // namespace tcvl
