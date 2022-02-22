#include <gflags/gflags.h>

#include <beam_utils/gflags.h>


int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);


  return 0;
}
