#include <TCVL/utils.h>

#include <beam_calibration/CameraModel.h>
#include <beam_calibration/TfTree.h>
#include <beam_mapping/Poses.h>
#include <beam_utils/filesystem.h>
#include <beam_utils/gflags.h>
#include <beam_utils/log.h>

#include <boost/filesystem.hpp>
#include <gflags/gflags.h>

DEFINE_string(config_file, "", "Full path to config file to load (Required).");
DEFINE_validator(config_file, &beam::gflags::ValidateFileMustExist);

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!boost::filesystem::exists(FLAGS_config_file)) {
    BEAM_ERROR("Invalid Config File Path.");
    return -1;
  }

  nlohmann::json J;
  beam::ReadJson(FLAGS_config_file, J);

  // Load image folder paths into ordered set
  std::string keyframe_dir = J["keyframe_dir"];
  if (!boost::filesystem::exists(keyframe_dir) ||
      !boost::filesystem::is_directory(keyframe_dir)) {
    BEAM_ERROR("Invalid Keyframe Directory.");
    return -1;
  }
  std::set<uint64_t> keyframe_timestamps =
      GetTimestampsFromFolder(keyframe_dir);

  // Load scan folder paths into ordered set
  std::string scans_dir = J["scans_dir"];
  if (!boost::filesystem::exists(scans_dir) ||
      !boost::filesystem::is_directory(scans_dir)) {
    BEAM_ERROR("Invalid Lidar Scans Directory.");
    return -1;
  }
  std::set<uint64_t> lidar_timestamps = GetTimestampsFromFolder(scans_dir);

  // Load poses object
  std::string poses_file = J["poses_file"];
  if (!boost::filesystem::exists(poses_file) ||
      boost::filesystem::is_directory(poses_file)) {
    BEAM_ERROR("Invalid Poses File Path.");
    return -1;
  }
  beam_mapping::Poses poses;
  poses.LoadFromPLY(poses_file);

  // Load robot extrinsics
  std::string extrinsics_file = J["extrinsics_file"];
  if (!boost::filesystem::exists(extrinsics_file) ||
      boost::filesystem::is_directory(extrinsics_file)) {
    BEAM_ERROR("Invalid Extrinsics File Path.");
    return -1;
  }
  beam_calibration::TfTree tree;
  tree.LoadJSON(extrinsics_file);

  // Load camera intrinsics
  std::string cam_intrinsics_file = J["cam_intrinsics_file"];
  if (!boost::filesystem::exists(cam_intrinsics_file) ||
      boost::filesystem::is_directory(poses_file)) {
    BEAM_ERROR("Invalid Camera Intrinsics File Path.");
    return -1;
  }
  std::shared_ptr<beam_calibration::CameraModel> cam_model =
      beam_calibration::CameraModel::Create(cam_intrinsics_file);

  return 0;
}
