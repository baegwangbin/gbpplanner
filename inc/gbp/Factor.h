/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed under MIT license (see LICENSE for details)
/**************************************************************************************/
#pragma once
#include "Simulator.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Utils.h>
#include <gbp/GBPCore.h>
#include <gbp/lie_algebra.h>
#include <gbp/Variable.h>
#include <manif/SE2.h>
#include <manif/SO2.h>
#include <raylib.h>
#include <tuple>
#include <typeinfo>
#include <vector>

extern Globals globals;

using Eigen::seqN;
using Eigen::seq;
using Eigen::last;



// class Variable;     // Forward declaration

// Types of factors defined. Default is DEFAULT_FACTOR
enum FactorType {DEFAULT_FACTOR, DYNAMICS_FACTOR, INTERROBOT_FACTOR, OBSTACLE_FACTOR};
/*****************************************************************************************/
// Factor used in GBP
/*****************************************************************************************/
class Factor {
    public:
    Simulator* sim_;                            // Pointer to simulator
    int f_id_;                                  // Factor id
    int r_id_;                                  // Robot id this factor belongs to
    Key key_;                                   // Factor key = {r_id_, f_id_}
    int other_rid_;                             // id of other connected robot (if this is an inter-robot factor)
    int n_dofs_;                                // n_dofs of the variables connected to this factor
    Eigen::VectorXd z_;                         // Measurement
    Eigen::MatrixXd h_, J_;                     // Stored values of measurement function h_func_() and Jacobian J_func_()
    Eigen::VectorXd X_;                         // Stored linearisation point
    Eigen::MatrixXd meas_model_lambda_;         // Precision of measurement model
    Mailbox inbox_, outbox_, last_outbox_;      
    FactorType factor_type_ = DEFAULT_FACTOR; 
    float delta_jac=1e-8;                       // Delta used for first order jacobian calculation
    bool initialised_ = false;                  // Becomes true when Jacobian calculated for the first time
    bool linear_ = false;                       // True is factor is linear (avoids recomputation of Jacobian)
    bool skip_flag = false;                          // Flag to skip factor update if required
    virtual bool skip_factor(){                 // Default function to set skip flag
        // skip_flag = false;
        return skip_flag;
    };
    double mahalanobis_threshold_ = 2.;
    double sigma_;
    bool robust_flag_ = false;
    double adaptive_gauss_noise_var_;
    std::vector<std::shared_ptr<Variable>> variables_{};    // Vector of pointers to the connected variables. Order of variables matters
    bool active_ = true;
    double damping_ = 0.;
    

    // Function declarations
    Factor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
            float sigma, const Eigen::VectorXd& measurement, 
            int n_dofs=4);

    ~Factor();

    void draw();

    virtual Eigen::MatrixXd h_func_(const Eigen::VectorXd& X) = 0;

    virtual Eigen::MatrixXd J_func_(const Eigen::VectorXd& X);

    Eigen::MatrixXd jacobianFirstOrder(const Eigen::VectorXd& X0);

    virtual Eigen::VectorXd residual(){return z_ - h_;};

    bool update_factor();

    Message marginalise_factor_dist(const Eigen::VectorXd &eta, const Eigen::MatrixXd &Lam, int var_idx, int marg_idx);
};


/********************************************************************************************/
/* Reprojection factor */
/*****************************************************************************************************/
class ReprojectionFactor: public Factor {
    public:
    Eigen::MatrixXd K_;

    ReprojectionFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
        float sigma, const Eigen::Vector2d& measurement, const Eigen::Matrix3d& K);

    // Constant velocity model
    Eigen::MatrixXd h_func_(const Eigen::VectorXd& X);
    Eigen::MatrixXd J_func_(const Eigen::VectorXd& X);

};

/********************************************************************************************/
/* Dynamics factor: constant-velocity model */
/*****************************************************************************************************/
class DynamicsFactor: public Factor {
    public:

    DynamicsFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
        float sigma, const Eigen::VectorXd& measurement, float dt);

    // Constant velocity model
    Eigen::MatrixXd h_func_(const Eigen::VectorXd& X);
    Eigen::MatrixXd J_func_(const Eigen::VectorXd& X);

};

/********************************************************************************************/
/* Interrobot factor: for avoidance of other robots */
// This factor results in a high energy or cost if two robots are planning to be in the same 
// position at the same timestep (collision). This factor is created between variables of two robots.
// The factor has 0 energy if the variables are further away than the safety distance. skip_ = true in this case.
/********************************************************************************************/
class InterrobotFactor: public Factor {
    public:
    double safety_distance_;

    InterrobotFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
        float sigma, const Eigen::VectorXd& measurement, 
        float robot_radius);

    Eigen::MatrixXd h_func_(const Eigen::VectorXd& X);
    Eigen::MatrixXd J_func_(const Eigen::VectorXd& X);
    bool skip_factor();

};

/********************************************************************************************/
// Obstacle factor for static obstacles in the scene. This factor takes a pointer to the obstacle image from the Simulator.
// Note. in the obstacle image, white areas represent obstacles (as they have a value of 1).
// The input image to the simulator is opposite, which is why it needs to be inverted.
// The delta used in the first order jacobian calculation is chosen such that it represents one pixel in the image.
/********************************************************************************************/
class ObstacleFactor: public Factor {
    public:
    Image* p_obstacleImage_;

    ObstacleFactor(Simulator* sim, int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
        float sigma, const Eigen::VectorXd& measurement, Image* p_obstacleImage);

    Eigen::MatrixXd h_func_(const Eigen::VectorXd& X);

};



//
class FactorLie {
    public:
    std::vector<LieType> lietypes_{};
    LieType measurement_lietype_;
    int f_id_;                                  // Factor id
    int r_id_;                                  // Robot id this factor belongs to
    Key key_;                                   // Factor key = {r_id_, f_id_}
    int other_rid_;                             // id of other connected robot (if this is an inter-robot factor)
    Eigen::MatrixXd meas_model_lambda_;         // Precision of measurement model
    std::vector<Eigen::VectorXd> lin_point_{}; 
    Eigen::VectorXd z_;    
    FactorType factor_type_ = DEFAULT_FACTOR; 
    bool initialised_ = false;                  // Becomes true when Jacobian calculated for the first time
    bool linear_ = false;                       // True is factor is linear (avoids recomputation of Jacobian)
    bool skip_flag = false;                          // Flag to skip factor update if required
    virtual bool skip_factor(){                 // Default function to set skip flag
        return skip_flag;
    };
    double mahalanobis_threshold_ = 2.;
    double sigma_;
    bool robust_flag_ = false;
    double adaptive_gauss_noise_var_;
    std::vector<std::shared_ptr<VariableLie>> variables_{};    // Vector of pointers to the connected variables. Order of variables matters
    bool active_ = true;
    double damping_ = 0.;

    MailboxLie inbox_, outbox_, last_outbox_;


    // Function declarations
    FactorLie(int f_id, int r_id, std::vector<std::shared_ptr<VariableLie>> variables,
            float sigma, Eigen::VectorXd measurement, LieType measurement_lietype);

    ~FactorLie();

    virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> computeResidualJacobian() = 0;
    // void draw();
    bool update_factor();

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> marginalise_factor_dist(const Eigen::VectorXd &eta, const Eigen::MatrixXd &Lam, int var_idx, int marg_idx);

};
/********************************************************************************************/
/* Prior SE2d factor */
/*****************************************************************************************************/
class PriorFactor: public FactorLie {
    public:
    int n_dofs_;

    PriorFactor(int f_id, int r_id, std::vector<std::shared_ptr<VariableLie>> variables,
            float sigma, Eigen::VectorXd measurement, LieType lietype);

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> computeResidualJacobian();

};
/********************************************************************************************/
/* Prior SE2d factor */
/*****************************************************************************************************/
class PriorFactorSE2_2d: public FactorLie {
    public:
    int n_dofs_;

    PriorFactorSE2_2d(int f_id, int r_id, std::vector<std::shared_ptr<VariableLie>> variables,
            float sigma, Eigen::VectorXd measurement, LieType lietype);

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> computeResidualJacobian();

};
/********************************************************************************************/
/* Smoothness factor */
/*****************************************************************************************************/
class SmoothnessFactor: public FactorLie {
    public:
    int n_dofs_;

    SmoothnessFactor(int f_id, int r_id, std::vector<std::shared_ptr<VariableLie>> variables,
            float sigma, Eigen::VectorXd measurement, LieType lietype);

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> computeResidualJacobian();

};
/********************************************************************************************/
/* Angle Difference SE2d factor */
/*****************************************************************************************************/
class AngleDifferenceFactorSE2d: public FactorLie {
    public:

    AngleDifferenceFactorSE2d(int f_id, int r_id, std::vector<std::shared_ptr<VariableLie>> variables,
            float sigma, Eigen::VectorXd measurement);

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> computeResidualJacobian();

};


/********************************************************************************************/
/* Dynamics factor: constant-velocity model */
/*****************************************************************************************************/
class DynamicsFactorSE2_2d: public FactorLie {
    public:
    float dt_;
    int n_dofs_;

    DynamicsFactorSE2_2d(int f_id, int r_id, std::vector<std::shared_ptr<VariableLie>> variables,
        float sigma, const Eigen::VectorXd& measurement, float dt);

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> computeResidualJacobian();

};
/********************************************************************************************/
/* Dynamics factor: constant-velocity model */
/*****************************************************************************************************/
class DynamicsFactorSE2_2d_test: public FactorLie {
    public:
    float dt_;
    int n_dofs_;

    DynamicsFactorSE2_2d_test(int f_id, int r_id, std::vector<std::shared_ptr<VariableLie>> variables,
        float sigma, const Eigen::VectorXd& measurement, float dt);

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> computeResidualJacobian();

};