#pragma once

#include <fuse_core/constraint.h>
#include <fuse_core/eigen.h>
#include <fuse_core/macros.h>
#include <fuse_core/serialization.h>
#include <fuse_core/uuid.h>
#include <fuse_variables/orientation_3d_stamped.h>
#include <fuse_variables/point_3d_landmark.h>
#include <fuse_variables/position_3d_stamped.h>

#include <beam_calibration/CameraModel.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>

#include <ostream>
#include <string>
#include <vector>

namespace fuse_constraints {

class ReprojectionConstraint : public fuse_core::Constraint {
public:
  FUSE_CONSTRAINT_DEFINITIONS(ReprojectionConstraint);

  /**
   * @brief Default constructor
   */
  ReprojectionConstraint() = default;

  /**
   * @brief Create a constraint using landmark location, camera pose and
   * measured pixel location
   *
   */
  ReprojectionConstraint(
      const std::string &source,
      const fuse_variables::Orientation3DStamped &R_WORLD_BASELINK,
      const fuse_variables::Position3DStamped &t_WORLD_BASELINK,
      const fuse_variables::Point3DLandmark &P_WORLD,
      const Eigen::Vector2d &pixel_measurement,
      const Eigen::Matrix4d &T_cam_baselink,
      const std::shared_ptr<beam_calibration::CameraModel> cam_model);

  /**
   * @brief Destructor
   */
  virtual ~ReprojectionConstraint() = default;

  /**
   * @brief Read-only access to the measured pixel value
   *
   */
  const Eigen::Vector2d &pixel() const { return pixel_; }

  /**
   * @brief Print a human-readable description of the constraint to the provided
   * stream.
   *
   * @param[out] stream The stream to write to. Defaults to stdout.
   */
  void print(std::ostream &stream = std::cout) const override;

  /**
   * @brief Construct an instance of this constraint's cost function
   *
   * The function caller will own the new cost function instance. It is the
   * responsibility of the caller to delete the cost function object when it is
   * no longer needed. If the pointer is provided to a Ceres::Problem object,
   * the Ceres::Problem object will takes ownership of the pointer and delete it
   * during destruction.
   *
   * @return A base pointer to an instance of a derived CostFunction.
   */
  ceres::CostFunction *costFunction() const override;

protected:
  Eigen::Vector2d pixel_;
  Eigen::Matrix4d T_cam_baselink_;
  std::shared_ptr<beam_calibration::CameraModel> cam_model_;

private:
  // Allow Boost Serialization access to private methods
  friend class boost::serialization::access;

  /**
   * @brief The Boost Serialize method that serializes all of the data members
   * in to/out of the archive
   *
   * @param[in/out] archive - The archive object that holds the serialized class
   * members
   * @param[in] version - The version of the archive being read/written.
   * Generally unused.
   */
  template <class Archive>
  void serialize(Archive &archive, const unsigned int /* version */) {
    archive &boost::serialization::base_object<fuse_core::Constraint>(*this);
    archive &pixel_;
  }
};

} // namespace fuse_constraints

BOOST_CLASS_EXPORT_KEY(fuse_constraints::ReprojectionConstraint);

// Temporary until .cpp and .h are needed