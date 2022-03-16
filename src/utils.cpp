#include <TCVL/utils.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <iostream>
#include <sstream>

namespace tcvl {

std::shared_ptr<beam_mapping::Poses> LoadPoses(const std::string &path) {
  std::shared_ptr<beam_mapping::Poses> poses =
      std::make_shared<beam_mapping::Poses>();
  // declare variables
  std::ifstream infile;
  std::string line;
  Eigen::Matrix4d Tk;
  ros::Time time_stamp_k;
  // open file
  infile.open(path);
  // extract poses
  while (!infile.eof()) {
    // get timestamp k
    std::getline(infile, line, ' ');
    if (line.length() > 0) {
      try {
        double t = std::stod(line);
        time_stamp_k = ros::Time(t);
      } catch (const std::invalid_argument &e) {
        BEAM_CRITICAL("Invalid argument, probably at end of file");
        throw std::invalid_argument{
            "Invalid argument, probably at end of file"};
      }

      Eigen::Vector3d p;
      Eigen::Quaterniond q;
      std::getline(infile, line, ' ');
      p[0] = std::stod(line);
      std::getline(infile, line, ' ');
      p[1] = std::stod(line);
      std::getline(infile, line, ' ');
      p[2] = std::stod(line);
      std::getline(infile, line, ' ');
      q.x() = std::stod(line);
      std::getline(infile, line, ' ');
      q.y() = std::stod(line);
      std::getline(infile, line, ' ');
      q.z() = std::stod(line);
      std::getline(infile, line, '\n');
      q.w() = std::stod(line);

      Eigen::Matrix4d Tk;
      beam::QuaternionAndTranslationToTransformMatrix(q, p, Tk);

      poses->AddSinglePose(Tk);
      poses->AddSingleTimeStamp(time_stamp_k);
    }
  }
  return poses;
}

} // namespace tcvl