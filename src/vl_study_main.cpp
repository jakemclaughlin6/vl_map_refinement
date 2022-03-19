#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>

#include <TCVL/lidar_visual_mapper.h>

#include <boost/foreach.hpp>
#include <gflags/gflags.h>

#define foreach BOOST_FOREACH

DEFINE_string(config_file, "", "Full path to config file to load (Required).");
DEFINE_validator(config_file, &beam::gflags::ValidateFileMustExist);

int main(int argc, char *argv[]) {
  // Load config file
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (!boost::filesystem::exists(FLAGS_config_file)) {
    BEAM_ERROR("Invalid Config File Path.");
    return -1;
  }
  nlohmann::json J;
  beam::ReadJson(FLAGS_config_file, J);

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
  std::shared_ptr<tcvl::LidarVisualMapper> mapper =
      std::make_shared<tcvl::LidarVisualMapper>(
          J["cam_intrinsics_file"], J["pose_file"], J["extrinsics_file"],
          J["camera_frame_id"], J["lidar_frame_id"], J["pose_frame_id"],
          start_time, end_time);

  // Begin processing loop
  foreach (rosbag::MessageInstance const m, view) {
    // add lidar scan to mapper
    sensor_msgs::PointCloud2::Ptr scan =
        m.instantiate<sensor_msgs::PointCloud2>();
    if (scan) {
      mapper->AddLidarScan(scan);
    }
    // process image
    sensor_msgs::CompressedImage::Ptr buffer_image_compressed =
        m.instantiate<sensor_msgs::CompressedImage>();
    if (buffer_image_compressed) {
      cv::Mat image =
          cv::imdecode(buffer_image_compressed->data, cv::IMREAD_GRAYSCALE);
      ros::Time stamp = buffer_image_compressed->header.stamp;
      mapper->ProcessImage(image, stamp);
    }
  }

  // optimize graph
  // compare results

  return 0;
}
