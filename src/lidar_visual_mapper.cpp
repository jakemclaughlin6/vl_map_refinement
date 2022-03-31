#include <fuse_constraints/absolute_pose_3d_stamped_constraint.h>
#include <fuse_constraints/relative_pose_3d_stamped_constraint.h>
#include <vl_map_refinement/constraints/reprojection_constraint.h>
#include <vl_map_refinement/constraints/vl_constraint.h>
#include <vl_map_refinement/lidar_visual_mapper.h>

#include <beam_cv/detectors/Detectors.h>
#include <beam_cv/geometry/Triangulation.h>
#include <beam_depth/DepthMap.h>
#include <beam_depth/Utils.h>
#include <beam_utils/pointclouds.h>
#include <boost/filesystem.hpp>

namespace vl_map_refinement {

LidarVisualMapper::LidarVisualMapper(
    const std::string &cam_model_path, const std::string &pose_lookup_path,
    const std::string &extrinsics_path, const std::string &camera_frame_id,
    const std::string &lidar_frame_id, const std::string &pose_frame_id,
    const ros::Time &start_time, const ros::Time &end_time,
    const double &odom_prior_covariance_diag)
    : camera_frame_id_(camera_frame_id), lidar_frame_id_(lidar_frame_id),
      pose_frame_id_(pose_frame_id), start_time_(start_time),
      end_time_(end_time) {

  // Load initial pose estimates
  if (!boost::filesystem::exists(pose_lookup_path) ||
      boost::filesystem::is_directory(pose_lookup_path)) {
    BEAM_ERROR("Invalid Poses File Path.");
    throw std::runtime_error{"Invalid Poses File Path."};
  }
  std::shared_ptr<beam_mapping::Poses> poses =
      std::make_shared<beam_mapping::Poses>();
  poses->LoadFromPLY(pose_lookup_path);
  pose_lookup_ = std::make_shared<vl_map_refinement::PoseLookup>(
      poses, pose_frame_id, "world");

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
      std::make_shared<beam_cv::FASTDetector>(200);
  tracker_ = std::make_shared<beam_cv::KLTracker>(detector, nullptr, 100);

  // create new graph object
  graph_ = std::make_shared<fuse_graphs::HashGraph>();

  // initialize covariance matrix for priors on poses
  prior_covariance_ =
      Eigen::Matrix<double, 6, 6>::Identity() * odom_prior_covariance_diag;
}

void LidarVisualMapper::AddLidarScan(
    const sensor_msgs::PointCloud2::Ptr &scan) {
  // transform into world frame to keep a running local point cloud
  pcl::PointCloud<pcl::PointXYZ> cloud = beam::ROSToPCL(*scan);

  // get point clouds pose
  Eigen::Matrix4d T_WORLD_BASELINK;
  pose_lookup_->GetT_WORLD_SENSOR(T_WORLD_BASELINK, scan->header.stamp);

  // perturb pose (for experiment)
  T_WORLD_BASELINK = PerturbPose(scan->header.stamp, T_WORLD_BASELINK);

  // get pose in lidar frame
  Eigen::Matrix4d T_WORLD_LIDAR = T_WORLD_BASELINK * T_baselink_lidar_;

  // transform scan into world frame
  pcl::PointCloud<pcl::PointXYZ> cloud_in_world_frame;
  pcl::transformPointCloud(cloud, cloud_in_world_frame, T_WORLD_LIDAR);

  current_clouds_.push_back(cloud_in_world_frame);

  if (current_clouds_.size() >= 4) {
    current_clouds_.pop_front();
  }
}

void LidarVisualMapper::ProcessImage(const cv::Mat &image,
                                     const ros::Time &timestamp) {
  // add image to tracker
  tracker_->AddImage(image, timestamp);

  // determine if its a keyframe
  if (previous_keyframes_.empty() ||
      timestamp.toSec() - previous_keyframes_.back().toSec() >= 0.2) {
    std::cout << "\nProcessing keyframe " << GetNumKeyframes() << ":"
              << std::endl;

    // push keyframe time to queue
    previous_keyframes_.push_back(timestamp);
    num_keyframes_++;

    // add keyframes pose to the graph
    Eigen::Matrix4d T_WORLD_BASELINK;
    pose_lookup_->GetT_WORLD_SENSOR(T_WORLD_BASELINK, timestamp);

    // perturb pose (for experiment)
    T_WORLD_BASELINK = PerturbPose(timestamp, T_WORLD_BASELINK);

    // add pose to graph
    AddBaselinkPose(T_WORLD_BASELINK, timestamp);

    size_t num_lms = 0;
    size_t num_rep_c = 0;
    // triangulate possible landmarks and add reproj constraints
    if (previous_keyframes_.size() > 3) {
      std::vector<uint64_t> landmarks =
          tracker_->GetLandmarkIDsInImage(timestamp);
      for (auto &id : landmarks) {
        fuse_variables::Point3DLandmark::SharedPtr lm = GetLandmark(id);
        if (lm) {
          // check if pixel can be undistorted
          Eigen::Vector2d pixel = tracker_->Get(timestamp, id);
          Eigen::Vector2i tmp;
          if (!cam_model_->UndistortPixel(pixel.cast<int>(), tmp))
            continue;

          AddReprojectionConstraint(timestamp, id, pixel);
          num_rep_c++;
        } else {
          // get measurements of landmark for triangulation
          std::vector<Eigen::Matrix4d, beam::AlignMat4d> T_cam_world_v;
          std::vector<Eigen::Vector2i, beam::AlignVec2i> pixels;
          std::vector<ros::Time> observation_stamps;

          for (auto &kf_time : previous_keyframes_) {
            try {
              // check if pixel can be undistorted
              Eigen::Vector2d pixel = tracker_->Get(kf_time, id);
              Eigen::Vector2i tmp;
              if (!cam_model_->UndistortPixel(pixel.cast<int>(), tmp))
                continue;

              Eigen::Matrix4d T_world_cam;
              if (GetCameraPose(kf_time, T_world_cam)) {
                pixels.push_back(pixel.cast<int>());
                T_cam_world_v.push_back(T_world_cam.inverse());
                observation_stamps.push_back(kf_time);
              }
            } catch (const std::out_of_range &oor) {
            }
          }
          // triangulate new points
          if (T_cam_world_v.size() >= 4) {
            beam::opt<Eigen::Vector3d> point =
                beam_cv::Triangulation::TriangulatePoint(
                    cam_model_, T_cam_world_v, pixels, 30, 5.0);
            if (point.has_value()) {
              AddLandmark(point.value(), id);
              num_lms++;
              for (int i = 0; i < observation_stamps.size(); i++) {
                num_rep_c++;
                AddReprojectionConstraint(
                    observation_stamps[i], id,
                    tracker_->Get(observation_stamps[i], id));
              }
            }
          }
        }
      }
    }

    // add relative pose constraint
    if (previous_keyframes_.size() >= 2) {
      AddRelativePoseConstraint(
          timestamp, previous_keyframes_[previous_keyframes_.size() - 2]);
    } else if (previous_keyframes_.size() == 1) {
      AddAbsolutePoseConstraint(timestamp);
    }

    std::cout << "Added " << num_lms << " visual landmarks." << std::endl;
    std::cout << "Added " << num_rep_c << " reprojection constraints."
              << std::endl;

    // add tightly coupled VL constraints
    ProcessLidarCoupling(timestamp);

    // manage keyframes
    if (previous_keyframes_.size() > 20) {
      previous_keyframes_.pop_front();
    }
  }
}

void LidarVisualMapper::ProcessLidarCoupling(const ros::Time &kf_time) {
  const int R = 7;
  Eigen::Matrix4d T_WORLD_CAM;
  if (GetCameraPose(kf_time, T_WORLD_CAM)) {
    size_t num_vl_c = 0;

    // aggregate current scans
    pcl::PointCloud<pcl::PointXYZ> aggregate_cloud;
    for (auto &c : current_clouds_) {
      aggregate_cloud += c;
    }

    // transform the current point cloud into the camera frame
    std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud_in_camera_frame =
        std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    Eigen::Matrix4d T_CAM_WORLD = T_WORLD_CAM.inverse();
    pcl::transformPointCloud(aggregate_cloud, *cloud_in_camera_frame,
                             T_CAM_WORLD);

    // extract depth map using projection
    beam_depth::DepthMap dm(cam_model_, cloud_in_camera_frame);
    dm.ExtractDepthMapProjection(30);
    cv::Mat depth_image = dm.GetDepthImage();

    // find visual-lidar correspondences
    std::vector<uint64_t> landmarks = tracker_->GetLandmarkIDsInImage(kf_time);
    for (auto &id : landmarks) {
      fuse_variables::Point3DLandmark::SharedPtr lm = GetLandmark(id);
      if (!lm)
        continue;

      Eigen::Vector2i pixel = tracker_->Get(kf_time, id).cast<int>();

      // compute search area
      int start_col = pixel[0] - R, end_col = pixel[0] + R,
          start_row = pixel[1] - R, end_row = pixel[1] + R;
      if (start_col < 0)
        start_col = 0;
      if (end_col > depth_image.cols)
        end_col = depth_image.cols;
      if (start_row < 0)
        start_row = 0;
      if (end_row > depth_image.cols)
        end_row = depth_image.cols;

      // vectors of pixels, pixel distances, depth
      std::vector<double> neighbourhood_depths;
      std::vector<double> neighbourhood_distances;
      std::vector<Eigen::Vector2i> neighbourhood_pixels;

      for (int col = start_col; col < end_col; col++) {
        for (int row = start_row; row < end_row; row++) {
          Eigen::Vector2i search_point(col, row);

          double depth = depth_image.at<float>(row, col);

          if (col < 0 || row < 0 || col > depth_image.cols ||
              row > depth_image.rows || depth == 0)
            continue;

          neighbourhood_depths.push_back(depth);
          neighbourhood_distances.push_back(
              beam::distance(pixel, search_point));
          neighbourhood_pixels.push_back(search_point);
        }
      }

      // 1. Remove any inconsistent points (10cm threshold)
      double sum = std::accumulate(std::begin(neighbourhood_depths),
                                   std::end(neighbourhood_depths), 0.0);
      double mean_depth = sum / neighbourhood_depths.size();
      std::vector<double> filtered_depths;
      std::vector<double> filtered_distances;
      std::vector<Eigen::Vector2i> filtered_pixels;
      for (int i = 0; i < neighbourhood_depths.size(); i++) {
        if (neighbourhood_depths[i] < mean_depth + 0.1 &&
            neighbourhood_depths[i] > mean_depth - 0.1) {
          filtered_depths.push_back(neighbourhood_depths[i]);
          filtered_distances.push_back(neighbourhood_distances[i]);
          filtered_pixels.push_back(neighbourhood_pixels[i]);
        }
      }

      // 2. find 3 closest points to the landmark
      if (filtered_distances.size() < 3)
        continue;
      std::vector<Eigen::Vector2i> matching_pixels;
      std::vector<double> matching_depths;
      for (int i = 0; i < 3; i++) {
        int min_index = std::min_element(filtered_distances.begin(),
                                         filtered_distances.end()) -
                        filtered_distances.begin();

        matching_pixels.push_back(filtered_pixels[min_index]);
        matching_depths.push_back(filtered_depths[min_index]);

        filtered_depths.erase(filtered_depths.begin() + min_index);
        filtered_distances.erase(filtered_distances.begin() + min_index);
        filtered_pixels.erase(filtered_pixels.begin() + min_index);
      }

      // 3. get matching points in camera frame
      std::vector<Eigen::Vector3d> matching_points;
      for (int i = 0; i < 3; i++) {
        Eigen::Vector3d direction;
        if (cam_model_->BackProject(matching_pixels[i], direction)) {
          Eigen::Vector3d coords = matching_depths[i] * direction.normalized();
          matching_points.push_back(coords);
        }
      }

      // compute a confidence for the measurement
      Eigen::Matrix4d T_WORLD_CAM;
      GetCameraPose(kf_time, T_WORLD_CAM);
      Eigen::Vector4d lm_p(lm->x(), lm->y(), lm->z(), 1);
      Eigen::Vector3d lm_cam = (T_WORLD_CAM.inverse() * lm_p).hnormalized();
      double lm_depth = lm_cam.norm();
      double plane_depth =
          (matching_points[0].norm() + matching_points[1].norm() +
           matching_points[2].norm()) /
          3;
      double confidence = 1 / (std::abs(lm_depth - plane_depth));

      // 4. add constraint to the graph
      fuse_constraints::VLConstraint::SharedPtr vl_constraint =
          fuse_constraints::VLConstraint::make_shared(
              "vl_map_refinement", *GetOrientation(kf_time),
              *GetPosition(kf_time), *lm, T_cam_baselink_, matching_points[0],
              matching_points[1], matching_points[2], confidence);
      graph_->addConstraint(vl_constraint);
      num_vl_c++;
    }
    std::cout << "Added " << num_vl_c << " visual-lidar constraints."
              << std::endl;
  }
}

void LidarVisualMapper::OptimizeGraph() {
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = 6;
  options.num_linear_solver_threads = 6;
  options.minimizer_type = ceres::TRUST_REGION;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.max_num_iterations = 30;
  graph_->optimize(options);
}

void LidarVisualMapper::OutputResults(const std::string &folder) {
  if (!boost::filesystem::is_directory(folder)) {
    BEAM_ERROR("Invalid output folder.");
    throw std::runtime_error{"Invalid output folder."};
  }

  pcl::PointCloud<pcl::PointXYZ> landmark_cloud;
  // create ordered map of poses keyed by timestamp
  std::map<double, std::vector<double>> poses;
  for (auto &var : graph_->getVariables()) {
    fuse_variables::Position3DStamped::SharedPtr p =
        fuse_variables::Position3DStamped::make_shared();

    fuse_variables::Orientation3DStamped::SharedPtr q =
        fuse_variables::Orientation3DStamped::make_shared();

    fuse_variables::Point3DLandmark::SharedPtr landmark =
        fuse_variables::Point3DLandmark::make_shared();

    if (var.type() == q->type()) {
      *q = dynamic_cast<const fuse_variables::Orientation3DStamped &>(var);
      auto position_uuid = fuse_core::uuid::generate(p->type(), q->stamp(),
                                                     fuse_core::uuid::NIL);
      *p = dynamic_cast<const fuse_variables::Position3DStamped &>(
          graph_->getVariable(position_uuid));

      std::vector<double> pose_vector{p->x(), p->y(), p->z(), q->w(),
                                      q->x(), q->y(), q->z()};
      poses[q->stamp().toSec()] = pose_vector;
    } else if (var.type() == landmark->type()) {
      *landmark = dynamic_cast<const fuse_variables::Point3DLandmark &>(var);
      pcl::PointXYZ point(landmark->x(), landmark->y(), landmark->z());
      landmark_cloud.points.push_back(point);
    }
  }
  beam::SavePointCloud<pcl::PointXYZ>(folder + "/landmarks.pcd", landmark_cloud,
                                      beam::PointCloudFileType::PCDBINARY);

  // output results
  std::ofstream outfile(folder + "/stamped_traj_estimate.txt");
  pcl::PointCloud<pcl::PointXYZRGB> frame_cloud;
  for (auto &pose : poses) {
    double timestamp = pose.first;
    std::vector<double> pose_vector = pose.second;
    // add pose to a file in the given folder
    Eigen::Vector3d position(pose_vector[0], pose_vector[1], pose_vector[2]);
    Eigen::Quaterniond orientation(pose_vector[3], pose_vector[4],
                                   pose_vector[5], pose_vector[6]);
    std::stringstream line;
    line << std::fixed;
    line << timestamp << " ";
    line << position[0] << " " << position[1] << " " << position[2] << " "
         << orientation.x() << " " << orientation.y() << " " << orientation.z()
         << " " << orientation.w() << std::endl;
    outfile << line.str();

    // add pose to frame cloud
    Eigen::Matrix4d T_WORLD_BASELINK;
    beam::QuaternionAndTranslationToTransformMatrix(orientation, position,
                                                    T_WORLD_BASELINK);
    frame_cloud = beam::AddFrameToCloud(frame_cloud, T_WORLD_BASELINK, 0.001);
  }
  beam::SavePointCloud<pcl::PointXYZRGB>(folder + "/frames.pcd", frame_cloud,
                                         beam::PointCloudFileType::PCDBINARY);
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
            "vl_map_refinement", *orientation, *position, *lm, pixel,
            T_cam_baselink_, cam_model_);
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

void LidarVisualMapper::AddRelativePoseConstraint(
    const ros::Time &cur_kf_time, const ros::Time &prev_kf_time) {
  fuse_variables::Position3DStamped::SharedPtr p1 = GetPosition(prev_kf_time);
  fuse_variables::Orientation3DStamped::SharedPtr o1 =
      GetOrientation(prev_kf_time);
  fuse_variables::Position3DStamped::SharedPtr p2 = GetPosition(cur_kf_time);
  fuse_variables::Orientation3DStamped::SharedPtr o2 =
      GetOrientation(cur_kf_time);

  // get pose 1 as eigen transform
  Eigen::Vector3d position1(p1->data());
  Eigen::Quaterniond orientation1(o1->w(), o1->x(), o1->y(), o1->z());
  Eigen::Matrix4d T_WORLD_FRAME1;
  beam::QuaternionAndTranslationToTransformMatrix(orientation1, position1,
                                                  T_WORLD_FRAME1);

  // get pose 2 as eigen transform
  Eigen::Vector3d position2(p2->data());
  Eigen::Quaterniond orientation2(o2->w(), o2->x(), o2->y(), o2->z());
  Eigen::Matrix4d T_WORLD_FRAME2;
  beam::QuaternionAndTranslationToTransformMatrix(orientation2, position2,
                                                  T_WORLD_FRAME2);

  // get relative transform between frames
  Eigen::Matrix4d T_FRAME1_FRAME2 = T_WORLD_FRAME1.inverse() * T_WORLD_FRAME2;
  Eigen::Vector3d rel_p;
  Eigen::Quaterniond rel_q;
  beam::TransformMatrixToQuaternionAndTranslation(T_FRAME1_FRAME2, rel_q,
                                                  rel_p);

  // create fuse constraint and add to graph
  fuse_core::Vector7d pose_relative_mean;
  pose_relative_mean << rel_p[0], rel_p[1], rel_p[2], rel_q.w(), rel_q.x(),
      rel_q.y(), rel_q.z();
  auto constraint =
      fuse_constraints::RelativePose3DStampedConstraint::make_shared(
          "vl_map_refinement", *p1, *o1, *p2, *o2, pose_relative_mean,
          prior_covariance_);
  graph_->addConstraint(constraint);
}

void LidarVisualMapper::AddAbsolutePoseConstraint(
    const ros::Time &cur_kf_time) {
  // set a covariance with near absolute certainty
  static Eigen::Matrix<double, 6, 6> anchor_covariance;
  anchor_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 0.0000000000001;
  // get position of keyframe
  fuse_variables::Position3DStamped::SharedPtr p = GetPosition(cur_kf_time);
  fuse_variables::Orientation3DStamped::SharedPtr o =
      GetOrientation(cur_kf_time);
  // create constraint and add to graph
  fuse_core::Vector7d mean;
  mean << p->x(), p->y(), p->z(), o->w(), o->x(), o->y(), o->z();
  auto constraint =
      fuse_constraints::AbsolutePose3DStampedConstraint::make_shared(
          "vl_map_refinement", *p, *o, mean, anchor_covariance);
  graph_->addConstraint(constraint);
}

Eigen::Matrix4d
LidarVisualMapper::PerturbPose(const ros::Time &stamp,
                               const Eigen::Matrix4d &T_WORLD_BASELINK) {
  // perturbation in roll, pitch, yaw, x, y, z (deg/s and m/s)
  Eigen::Matrix<double, 6, 1> perturbation;
  perturbation << 0, 0, 0.5, 0.02, 0.02, 0.001;

  // compute true perturbation based on how much time has passed
  double time_since_start = stamp.toSec() - start_time_.toSec();
  Eigen::Matrix<double, 6, 1> weighted_perturbation =
      perturbation * time_since_start;

  // perturb pose
  return beam::PerturbTransformDegM(T_WORLD_BASELINK, weighted_perturbation);
}

} // namespace vl_map_refinement
