#include <TCVL/utils.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <iostream>
#include <sstream>

std::set<uint64_t> GetTimestampsFromFolder(const std::string &path) {

  std::set<uint64_t> out_set;

  boost::filesystem::path p(path);

  boost::filesystem::directory_iterator end_itr;

  // cycle through the directory
  for (boost::filesystem::directory_iterator itr(p); itr != end_itr; ++itr) {

    if (boost::filesystem::is_regular_file(itr->path())) {
      std::string file = itr->path().stem().string();
      file.erase(std::remove(file.begin(), file.end(), '.'), file.end());
      uint64_t timestamp;
      std::istringstream iss(file);
      iss >> timestamp;
      out_set.insert(timestamp);
    }
  }

  return out_set;
}