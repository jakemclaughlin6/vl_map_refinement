#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <vl_map_refinement/lidar_visual_mapper.h>

#include <beam_cv/OpenCVConversions.h>

#include <boost/foreach.hpp>
#include <gflags/gflags.h>

#define foreach BOOST_FOREACH

DEFINE_string(config_file, "", "Full path to config file to load (Required).");
DEFINE_validator(config_file, &beam::gflags::ValidateFileMustExist);

bool ValidateJsonConfig(const nlohmann::json &J);

int main(int argc, char *argv[]) {
  // Load config file
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (!boost::filesystem::exists(FLAGS_config_file)) {
    BEAM_ERROR("Invalid Config File Path.");
    return -1;
  }
  nlohmann::json J;
  beam::ReadJson(FLAGS_config_file, J);
  if (!ValidateJsonConfig(J)) {
    return -1;
  }

  // Load bag file
  std::string bag_file = J["bag_file"];
  if (!boost::filesystem::exists(bag_file) ||
      boost::filesystem::is_directory(bag_file)) {
    BEAM_ERROR("Invalid Bag File Path.");
    return -1;
  }
  rosbag::Bag bag;
  bag.open(bag_file);

  // Open bag and get its info
  std::vector<std::string> topics{J["image_topic"], J["lidar_topic"]};
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  ros::Time start_time = view.getBeginTime();
  ros::Time end_time = view.getEndTime();

  // Initialize lidar-visual mapper to do the bulk of the work
  std::shared_ptr<vl_map_refinement::LidarVisualMapper> mapper =
      std::make_shared<vl_map_refinement::LidarVisualMapper>(
          J["cam_intrinsics_file"], J["pose_file"], J["extrinsics_file"],
          J["camera_frame_id"], J["lidar_frame_id"], J["pose_frame_id"],
          start_time, end_time);

  int max_keyframes = J["max_keyframes"];
  // Begin processing loop
  foreach (rosbag::MessageInstance const m, view) {
    // add lidar scan to mapper
    sensor_msgs::PointCloud2::Ptr scan =
        m.instantiate<sensor_msgs::PointCloud2>();
    if (scan) {
      mapper->AddLidarScan(scan);
    }

    // process image
    sensor_msgs::Image::Ptr buffer_image = m.instantiate<sensor_msgs::Image>();
    if (buffer_image) {
      cv::Mat image = beam_cv::OpenCVConversions::RosImgToMat(*buffer_image);
      ros::Time stamp = buffer_image->header.stamp;
      mapper->ProcessImage(image, stamp);
    }

    if (mapper->GetNumKeyframes() >= max_keyframes) {
      mapper->OptimizeGraph();
      mapper->OutputResults(J["output_folder"]);
      break;
    }
  }

  return 0;
}

bool ValidateJsonConfig(const nlohmann::json &J) {
  bool pass = true;
  if (J.find("bag_file") == J.end()) {
    BEAM_ERROR("Field 'bag_file' missing.");
    pass = false;
  }

  if (J.find("image_topic") == J.end()) {
    BEAM_ERROR("Field 'image_topic' missing.");
    pass = false;
  }

  if (J.find("lidar_topic") == J.end()) {
    BEAM_ERROR("Field 'lidar_topic' missing.");
    pass = false;
  }

  if (J.find("cam_intrinsics_file") == J.end()) {
    BEAM_ERROR("Field 'cam_intrinsics_file' missing.");
    pass = false;
  }
  if (J.find("pose_file") == J.end()) {
    BEAM_ERROR("Field 'pose_file' missing.");
    pass = false;
  }
  if (J.find("extrinsics_file") == J.end()) {
    BEAM_ERROR("Field 'extrinsics_file' missing.");
    pass = false;
  }
  if (J.find("camera_frame_id") == J.end()) {
    BEAM_ERROR("Field 'camera_frame_id' missing.");
    pass = false;
  }
  if (J.find("lidar_frame_id") == J.end()) {
    BEAM_ERROR("Field 'lidar_frame_id' missing.");
    pass = false;
  }
  if (J.find("pose_frame_id") == J.end()) {
    BEAM_ERROR("Field 'pose_frame_id' missing.");
    pass = false;
  }
  if (J.find("max_keyframes") == J.end()) {
    BEAM_ERROR("Field 'max_keyframes' missing.");
    pass = false;
  }
  if (J.find("output_folder") == J.end()) {
    BEAM_ERROR("Field 'output_folder' missing.");
    pass = false;
  }

  return pass;
}
