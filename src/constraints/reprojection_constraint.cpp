#include <vl_map_refinement/constraints/reprojection_constraint.h>

#include <vl_map_refinement/constraints/reprojection_functor.h>

#include <fuse_loss/huber_loss.h>

#include <pluginlib/class_list_macros.h>

#include <boost/serialization/export.hpp>

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace fuse_constraints {

ReprojectionConstraint::ReprojectionConstraint(
    const std::string &source,
    const fuse_variables::Orientation3DStamped &R_WORLD_BASELINK,
    const fuse_variables::Position3DStamped &t_WORLD_BASELINK,
    const fuse_variables::Point3DLandmark &P_WORLD,
    const Eigen::Vector2d &pixel_measurement,
    const Eigen::Matrix4d &T_cam_baselink,
    const std::shared_ptr<beam_calibration::CameraModel> cam_model)
    : fuse_core::Constraint(source, {R_WORLD_BASELINK.uuid(),
                                     t_WORLD_BASELINK.uuid(), P_WORLD.uuid()}) {
  pixel_ = pixel_measurement;
  cam_model_ = cam_model;
  T_cam_baselink_ = T_cam_baselink;

  fuse_loss::HuberLoss::SharedPtr l = std::make_shared<fuse_loss::HuberLoss>();
  loss(l);
}

void ReprojectionConstraint::print(std::ostream &stream) const {
  stream << type() << "\n"
         << "  source: " << source() << "\n"
         << "  uuid: " << uuid() << "\n"
         << "  pixel: " << pixel().transpose() << "\n";
}

ceres::CostFunction *ReprojectionConstraint::costFunction() const {
  return new ceres::AutoDiffCostFunction<ReprojectionFunctor, 2, 4, 3, 3>(
      new ReprojectionFunctor(pixel_, cam_model_, T_cam_baselink_));
}

} // namespace fuse_constraints
