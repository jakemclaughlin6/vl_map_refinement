#include <vl_map_refinement/constraints/vl_constraint.h>

#include <vl_map_refinement/constraints/point_to_plane_functor.h>

#include <fuse_loss/huber_loss.h>

#include <pluginlib/class_list_macros.h>

#include <boost/serialization/export.hpp>

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace fuse_constraints {

VLConstraint::VLConstraint(
    const std::string &source,
    const fuse_variables::Orientation3DStamped &R_WORLD_BASELINK,
    const fuse_variables::Position3DStamped &t_WORLD_BASELINK,
    const fuse_variables::Point3DLandmark &P_WORLD,
    const Eigen::Matrix4d &T_cam_baselink, const Eigen::Vector3d &P_REF1,
    const Eigen::Vector3d &P_REF2, const Eigen::Vector3d &P_REF3,
    const double &confidence)
    : fuse_core::Constraint(source, {R_WORLD_BASELINK.uuid(),
                                     t_WORLD_BASELINK.uuid(), P_WORLD.uuid()}) {

  T_cam_baselink_ = T_cam_baselink;
  P_REF1_ = P_REF1;
  P_REF2_ = P_REF2;
  P_REF3_ = P_REF3;
  confidence_ = confidence;

  fuse_loss::HuberLoss::SharedPtr l = std::make_shared<fuse_loss::HuberLoss>();
  loss(l);
}

void VLConstraint::print(std::ostream &stream) const {
  stream << type() << "\n"
         << "  source: " << source() << "\n"
         << "  uuid: " << uuid() << "\n"
         << "  P_REF1: " << P_REF1().transpose() << "\n"
         << "  P_REF2: " << P_REF2().transpose() << "\n"
         << "  P_REF3: " << P_REF3().transpose() << "\n";
}

ceres::CostFunction *VLConstraint::costFunction() const {
  return new ceres::AutoDiffCostFunction<PointToPlaneCostFunctor, 1, 4, 3, 3>(
      new PointToPlaneCostFunctor(P_REF1_, P_REF2_, P_REF3_, T_cam_baselink_,
                                  confidence_));
}

} // namespace fuse_constraints
