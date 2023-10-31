/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed under MIT license (see LICENSE for details)
/**************************************************************************************/
#pragma once

#include <limits>
#include <Eigen/Dense>
#include <Eigen/Core>
#define _EPS std::numeric_limits<double>::epsilon();
// Lie Algebra for S03
inline Eigen::Matrix3d S03_hat_operator(const Eigen::Vector3d& x){
    /// Hat operator for SO(3) Lie Group
    return Eigen::Matrix3d {{0., -x(2), x(1)},
                             {x(2), 0., -x(0)},
                             {-x(1), x(0), 0.}};
};

inline Eigen::Matrix3d so3exp(const Eigen::Vector3d& w){
    // Maps so(3) --> SO(3) group with closed form expression.
    double theta = w.norm();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity(3,3);
    if (theta < std::numeric_limits<double>::epsilon() * 3){
        return I;
    } else {
        Eigen::Matrix3d w_hat = S03_hat_operator(w);
        Eigen::Matrix3d R = I + (sin(theta) / theta) * w_hat + ((1 - cos(theta)) / pow(theta, 2.)) * (w_hat * w_hat);
        return R;
    }
};

inline Eigen::Vector3d so3log(const Eigen::Matrix3d& R){
    // Maps SO(3) --> so(3) group. Holds for d between -1 and 1
    if (R.isIdentity()){
        return Eigen::Vector3d{{0.,0.,0.}};
    } else {
        double d = 0.5 * (R.trace() - 1.);
        Eigen::Matrix3d lnR = (acos(d) / (2 * sqrt(1 - d*d))) * (R - R.transpose());

        Eigen::Vector3d w {{lnR(2,1), lnR(0,2), lnR(1,0)}};

        return w;        
    }
};

// DERIVATIVES

inline Eigen::MatrixXd dR_wx_dw(const Eigen::VectorXd& w, const Eigen::VectorXd& x){
    // """
    // :param w: Minimal SO(3) rep
    // :param x: 3D point / vector
    // :return: derivative of R(w)x wrt w
    // """
    Eigen::MatrixXd R = so3exp(w);
    Eigen::MatrixXd dR_wx_dw = - (R * S03_hat_operator(x)) *
                ( (w * w.transpose()) + (R.transpose() - Eigen::MatrixXd::Identity(3,3)) * S03_hat_operator(w) ) / w.squaredNorm();
    return dR_wx_dw;
};

inline Eigen::MatrixXd proj_derivative(const Eigen::VectorXd& x){
    Eigen::MatrixXd projected = Eigen::MatrixXd::Zero(2,3);
    projected(Eigen::all, Eigen::seqN(0,2)) = Eigen::MatrixXd::Identity(x.size()-1, x.size()-1) / x(Eigen::last);
    projected(Eigen::all, 2) = - x({0,1}) / pow(x(2), 2.);
    return projected;
};