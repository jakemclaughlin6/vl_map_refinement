#pragma once

#include <beam_mapping/Poses.h>
#include <map>
#include <set>
#include <string>

namespace tcvl {

std::shared_ptr<beam_mapping::Poses> LoadPoses(const std::string &path);
}