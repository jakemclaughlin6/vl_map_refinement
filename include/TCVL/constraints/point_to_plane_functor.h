#ifndef FUSE_MODELS_REPROJECTION_COST_FUNCTOR_H
#define FUSE_MODELS_REPROJECTION_COST_FUNCTOR_H

#include <fuse_core/eigen.h>
#include <fuse_core/macros.h>
#include <fuse_core/util.h>

#include <beam_cv/Utils.h>
#include <beam_utils/math.h>

#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function_to_functor.h>
#include <ceres/numeric_diff_cost_function.h>
#include <ceres/rotation.h>
namespace fuse_constraints {

class PointToPlaneCostFunctor {
public:
  FUSE_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Construct a cost function for a point to plane error
   * @param P_REF1 reference surface point 1 (in camera frame)
   * @param P_REF2 reference surface point 2 (in camera frame)
   * @param P_REF3 reference surface point 2 (in camera frame)
   * @param T_cam_baselink transform from baselink frame to camera frame
   */
  PointToPlaneCostFunctor(const Eigen::Vector3d &P_REF1,
                          const Eigen::Vector3d &P_REF2,
                          const Eigen::Vector3d &P_REF3,
                          const Eigen::Matrix4d &T_cam_baselink)
      : P_REF1_(P_REF1), P_REF2_(P_REF2), P_REF3_(P_REF3),
        T_cam_baselink_(T_cam_baselink) {}

  template <typename T>
  bool operator()(const T *const R_WORLD_BASELINK,
                  const T *const t_WORLD_BASELINK, const T *const P_WORLD,
                  T *residual) const {

    // transform point from world frame into camera frame
    Eigen::Matrix<T, 4, 4> T_CAM_BASELINK = T_cam_baselink_.cast<T>();

    T R_WORLD_BASELINK_mat[9];
    ceres::QuaternionToRotation(R_WORLD_BASELINK, R_WORLD_BASELINK_mat);

    Eigen::Matrix<T, 4, 4> T_WORLD_BASELINK;
    T_WORLD_BASELINK(0, 0) = R_WORLD_BASELINK_mat[0];
    T_WORLD_BASELINK(0, 1) = R_WORLD_BASELINK_mat[1];
    T_WORLD_BASELINK(0, 2) = R_WORLD_BASELINK_mat[2];
    T_WORLD_BASELINK(0, 3) = t_WORLD_BASELINK[0];
    T_WORLD_BASELINK(1, 0) = R_WORLD_BASELINK_mat[3];
    T_WORLD_BASELINK(1, 1) = R_WORLD_BASELINK_mat[4];
    T_WORLD_BASELINK(1, 2) = R_WORLD_BASELINK_mat[5];
    T_WORLD_BASELINK(1, 3) = t_WORLD_BASELINK[1];
    T_WORLD_BASELINK(2, 0) = R_WORLD_BASELINK_mat[6];
    T_WORLD_BASELINK(2, 1) = R_WORLD_BASELINK_mat[7];
    T_WORLD_BASELINK(2, 2) = R_WORLD_BASELINK_mat[8];
    T_WORLD_BASELINK(2, 3) = t_WORLD_BASELINK[2];
    T_WORLD_BASELINK(3, 0) = (T)0;
    T_WORLD_BASELINK(3, 1) = (T)0;
    T_WORLD_BASELINK(3, 2) = (T)0;
    T_WORLD_BASELINK(3, 3) = (T)1;

    Eigen::Matrix<T, 4, 1> P_WORLD_h;
    P_WORLD_h[0] = P_WORLD[0];
    P_WORLD_h[1] = P_WORLD[1];
    P_WORLD_h[2] = P_WORLD[2];
    P_WORLD_h[3] = (T)1;

    Eigen::Matrix<T, 4, 1> P_BASELINK_h =
        T_WORLD_BASELINK.inverse() * P_WORLD_h;
    Eigen::Matrix<T, 3, 1> P_CAM =
        (T_CAM_BASELINK * P_BASELINK_h).hnormalized();
    T P_CAMERA[3];
    P_CAMERA[0] = P_CAM[0];
    P_CAMERA[1] = P_CAM[1];
    P_CAMERA[2] = P_CAM[2];

    // cast plane member variables
    Eigen::Matrix<T, 3, 1> _P_REF1 = P_REF1_.cast<T>();
    Eigen::Matrix<T, 3, 1> _P_REF2 = P_REF2_.cast<T>();
    Eigen::Matrix<T, 3, 1> _P_REF3 = P_REF3_.cast<T>();

    /*
     * e = distance from point to line
     *   = | (P_REF - P_REF1) (P_REF1 - P_REF2) x (P_REF1 - P_REF3) |
     *     ----------------------------------------------------------
     *             | (P_REF1 - P_REF2) x (P_REF1 - P_REF3) |
     *
     *   = | dR1 (d12 x d12) |
     *     -------------------
     *       | d12 x d13 |
     *
     *   Where P_REF = T_REF_TGT * P_TGT
     *
     */
    T d12[3], d13[3], dR1[3];
    d12[0] = _P_REF1[0] - _P_REF2[0];
    d12[1] = _P_REF1[1] - _P_REF2[1];
    d12[2] = _P_REF1[2] - _P_REF2[2];

    d13[0] = _P_REF1[0] - _P_REF3[0];
    d13[1] = _P_REF1[1] - _P_REF3[1];
    d13[2] = _P_REF1[2] - _P_REF3[2];

    dR1[0] = P_CAMERA[0] - _P_REF1[0];
    dR1[1] = P_CAMERA[1] - _P_REF1[1];
    dR1[2] = P_CAMERA[2] - _P_REF1[2];

    T cross[3];
    ceres::CrossProduct(d12, d13, cross);
    T norm =
        sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);

    residual[0] = ceres::DotProduct(dR1, cross) / norm;

    return true;
  }

private:
  Eigen::Vector3d P_REF1_;
  Eigen::Vector3d P_REF2_;
  Eigen::Vector3d P_REF3_;
  Eigen::Matrix4d T_cam_baselink_;
};

} // namespace fuse_constraints

#endif // FUSE_MODELS_REPROJECTION_COST_FUNCTOR_H
