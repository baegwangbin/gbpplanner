/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed under MIT license (see LICENSE for details)
/**************************************************************************************/
#include <Utils.h>
#include <gbp/GBPCore.h>
#include <gbp/Factor.h>
#include <gbp/Variable.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <raylib.h>

/*****************************************************************************************************/
// Factor constructor
// Inputs:
//  - factor id (taken from simulator->next_f_id_++)
//  - robot id that this factor belongs to.
//  - A vector of pointers to Variables that the factor is to be connected to. Note, the order of the variables matters.
//  - sigma: factor strength. The factor precision Lambda = sigma^-2 * Identity
//  - measurement z: Eigen::VectorXd, must be same size as the output of the measurement function h().
//  - n_dofs is the number of degrees of freedom of the variables this factor is connected to. (eg. 4 for [x,y,xdot,ydot])
/*****************************************************************************************************/
Factor::Factor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
        float sigma, const Eigen::VectorXd& measurement, 
        int n_dofs) 
        : f_id_(f_id), r_id_(r_id), key_(r_id, f_id), variables_(variables), z_(measurement), n_dofs_(n_dofs), sigma_(sigma) {

        // Initialise precision of the measurement function
        this->meas_model_lambda_ = Eigen::MatrixXd::Identity(z_.rows(), z_.rows()) / pow(sigma_, 2.);
        this->adaptive_gauss_noise_var_ = pow(sigma_, 2.);
        
        // Initialise empty inbox and outbox
        int n_dofs_total = 0; int n_dofs_var;
        for (auto var : variables_) {
            n_dofs_var = var->n_dofs_;
            Message zero_msg(n_dofs_var);
            inbox_[var->key_] = zero_msg;
            outbox_[var->key_] = zero_msg;
            n_dofs_total += n_dofs_var;
        }

        // This parameter useful if the factor is connected to another robot
        other_rid_=r_id_;                           

        // Initialise empty linearisation point
        X_ = Eigen::VectorXd::Zero(n_dofs_total);
    };

/*****************************************************************************************************/
// Destructor
/*****************************************************************************************************/
Factor::~Factor(){
}

/*****************************************************************************************************/
// Drawing function for the factor. Draws a 3d Cylinder (line-ish) between its connected variables
/*****************************************************************************************************/
void Factor::draw(){
    if ((factor_type_==DYNAMICS_FACTOR && globals.DRAW_PATH)){
        auto v_0 = variables_[0];
        auto v_1 = variables_[1];
        if (!v_0->valid_ || !v_1->valid_) {return;};
        DrawCylinderEx(Vector3{(float)v_0->mu_(0), globals.ROBOT_RADIUS, (float)v_0->mu_(1)},
                        Vector3{(float)v_1->mu_(0), globals.ROBOT_RADIUS, (float)v_1->mu_(1)}, 
                        0.1, 0.1, 4, BLACK);        
    }    
}

/*****************************************************************************************************/
// Default measurement function h_func_() is the identity function: it returns the variable.
/*****************************************************************************************************/
Eigen::MatrixXd h_func_(const Eigen::VectorXd& X){return X;};
/*****************************************************************************************************/
// Default measurement function Jacobian J_func_() is the first order taylor series jacobian by default.
// When defining new factors, custom h_func_() and J_func_() must be defined, otherwise defaults are used.
/*****************************************************************************************************/
Eigen::MatrixXd Factor::J_func_(const Eigen::VectorXd& X){return this->jacobianFirstOrder(X);};

Eigen::MatrixXd Factor::jacobianFirstOrder(const Eigen::VectorXd& X0){
    Eigen::MatrixXd h0 = h_func_(X0);    // Value at lin point
    Eigen::MatrixXd jac_out = Eigen::MatrixXd::Zero(h0.size(),X0.size());
    for (int i=0; i<X0.size(); i++){
        Eigen::VectorXd X_copy = X0;                                    // Copy of lin point
        X_copy(i) += delta_jac;                                         // Perturb by delta
        jac_out(Eigen::all, i) = (h_func_(X_copy) - h0) / delta_jac;    // Derivative (first order)
    }
    return jac_out;
};

/*****************************************************************************************************/
// Main section: Factor update:
// Messages from connected variables are aggregated. The beliefs are used to create the linearisation point X_.
// The Factor potential is calculated using h_func_ and J_func_
// The factor precision and information is created, and then marginalised to create outgoing messages to its connected variables.
/*****************************************************************************************************/
bool Factor::update_factor(){

    // *Depending on the problem*, we may need to skip computation of this factor.
    // eg. to avoid extra computation, factor may not be required if two connected variables are too far apart.
    // in which case send out a Zero Message.
    if (this->skip_factor()){
        for (auto var : variables_){
            this->outbox_[var->key_] = Message(var->n_dofs_);
        }           
        return false;
    }

    // Messages from connected variables are aggregated.
    // The beliefs are used to create the linearisation point X_.
    int idx = 0; int n_dofs;
    for (int v=0; v<variables_.size(); v++){
        n_dofs = variables_[v]->n_dofs_;
        auto& [_, __, mu_belief] = this->inbox_[variables_[v]->key_];
        X_(seqN(idx, n_dofs)) = mu_belief;
        idx += n_dofs;
    }
    
    // The Factor potential and linearised Factor Precision and Information is calculated using h_func_ and J_func_
    // residual() is by default (z - h_func_(X))
    // Skip calculation of Jacobian if the factor is linear and Jacobian has already been computed once
    h_ = h_func_(X_);
    J_ = (this->linear_ && this->initialised_)? J_ : this->J_func_(X_);
    this->initialised_ = true;
    Eigen::MatrixXd factor_lam_potential = J_.transpose() * meas_model_lambda_ * J_;
    Eigen::VectorXd factor_eta_potential = (J_.transpose() * meas_model_lambda_) * (J_ * X_ + residual());

    /**** Robustify with HUBER loss *****/
    double robustness_k = 1.;
    double mahalanobis_dist = residual().norm() / this->sigma_;
    if (mahalanobis_dist >= mahalanobis_threshold_){
        robustness_k = (2. * mahalanobis_threshold_ * mahalanobis_dist - pow(mahalanobis_threshold_, 2.)) / pow(mahalanobis_dist, 2.);
    }

    factor_eta_potential *= robustness_k;
    factor_lam_potential *= robustness_k;


    //  Update factor precision and information with incoming messages from connected variables.
    int marginalisation_idx = 0;
    for (int v_out_idx=0; v_out_idx<variables_.size(); v_out_idx++){
        auto var_out = variables_[v_out_idx];
        // Initialise with factor values
        Eigen::VectorXd factor_eta = factor_eta_potential;     
        Eigen::MatrixXd factor_lam = factor_lam_potential;
        
        // Combine the factor with the belief from other variables apart from the receiving variable
        int idx_v = 0;
        for (int v_idx=0; v_idx<variables_.size(); v_idx++){
            int n_dofs = variables_[v_idx]->n_dofs_;
            if (variables_[v_idx]->key_ != var_out->key_) {
                auto [eta_belief, lam_belief, _] = inbox_[variables_[v_idx]->key_];
                factor_eta(seqN(idx_v, n_dofs)) += eta_belief;
                factor_lam(seqN(idx_v, n_dofs), seqN(idx_v, n_dofs)) += lam_belief;
            }
            idx_v += n_dofs;
        }
        
        // Marginalise the Factor Precision and Information to send to the relevant variable
        Message outgoing_msg = marginalise_factor_dist(factor_eta, factor_lam, v_out_idx, marginalisation_idx);

        // Apply damping and send out
        outgoing_msg.eta *= (1. - damping_);
        outgoing_msg.lambda *= (1. - damping_);
        outbox_[var_out->key_] = outgoing_msg;

        marginalisation_idx += var_out->n_dofs_;
    }

    return true;
};

/*****************************************************************************************************/
// Marginalise the factor Precision and Information and create the outgoing message to the variable
/*****************************************************************************************************/
Message Factor::marginalise_factor_dist(const Eigen::VectorXd &eta, const Eigen::MatrixXd &Lam, int var_idx, int marg_idx){
    // Marginalisation only needed if factor is connected to >1 variables
    int n_dofs = variables_[var_idx]->n_dofs_;
    if (eta.size() == n_dofs) return Message {eta, Lam};

    Eigen::VectorXd eta_a(n_dofs), eta_b(eta.size()-n_dofs);
    eta_a = eta(seqN(marg_idx, n_dofs));
    eta_b << eta(seq(0, marg_idx - 1)), eta(seq(marg_idx + n_dofs, last));

    Eigen::MatrixXd lam_aa(n_dofs, n_dofs), lam_ab(n_dofs, Lam.cols()-n_dofs);
    Eigen::MatrixXd lam_ba(Lam.rows()-n_dofs, n_dofs), lam_bb(Lam.rows()-n_dofs, Lam.cols()-n_dofs);
    lam_aa << Lam(seqN(marg_idx, n_dofs), seqN(marg_idx, n_dofs));
    lam_ab << Lam(seqN(marg_idx, n_dofs), seq(0, marg_idx - 1)), Lam(seqN(marg_idx, n_dofs), seq(marg_idx + n_dofs, last));
    lam_ba << Lam(seq(0, marg_idx - 1), seq(marg_idx, marg_idx + n_dofs - 1)), Lam(seq(marg_idx + n_dofs, last), seqN(marg_idx, n_dofs));
    lam_bb << Lam(seq(0, marg_idx - 1), seq(0, marg_idx - 1)), Lam(seq(0, marg_idx - 1), seq(marg_idx + n_dofs, last)),
            Lam(seq(marg_idx + n_dofs, last), seq(0, marg_idx - 1)), Lam(seq(marg_idx + n_dofs, last), seq(marg_idx + n_dofs, last));

    Eigen::MatrixXd lam_bb_inv = lam_bb.inverse();
    Message marginalised_msg(n_dofs);
    marginalised_msg.eta = eta_a - lam_ab * lam_bb_inv * eta_b;
    marginalised_msg.lambda = lam_aa - lam_ab * lam_bb_inv * lam_ba;
    if (!marginalised_msg.lambda.allFinite()) marginalised_msg.setZero();

    return marginalised_msg;
};    

/********************************************************************************************/
/********************************************************************************************/
//                      CUSTOM FACTORS SPECIFIC TO THE PROBLEM
// Create a new factor definition as shown with these examples.
// You may create a new factor_type_, in the enum in Factor.h (optional, default type is DEFAULT_FACTOR)
// Create a measurement function h_func_() and optionally Jacobian J_func_().

/********************************************************************************************/
/* Reprojection factor: */
/*****************************************************************************************************/
ReprojectionFactor::ReprojectionFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
    float sigma, const Eigen::Vector2d& measurement, const Eigen::Matrix3d& K)
    : Factor{f_id, r_id, variables, sigma, measurement}{ 
        factor_type_ = DEFAULT_FACTOR;
        // linear_ = true;
        damping_ = globals.DAMPING;
        K_ = K; // Intrinsic

    };

Eigen::MatrixXd ReprojectionFactor::h_func_(const Eigen::VectorXd& X){
    Eigen::VectorXd h = Eigen::VectorXd::Zero(2);
    
    Eigen::Vector3d t = X(seqN(0,3));
    Eigen::Matrix3d R_cw = so3exp(X(seqN(3,3)));
    Eigen::Vector3d y_wf = X(seqN(6,3));  

    Eigen::Vector3d reproj_temp = K_ * (R_cw * y_wf + t);
    h = reproj_temp({0,1}) / reproj_temp(2);
    return h;
}    
Eigen::MatrixXd ReprojectionFactor::J_func_(const Eigen::VectorXd& X){
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2,9);
    
    Eigen::Vector3d t = X(seqN(0,3));
    Eigen::Vector3d w = X(seqN(3,3));
    Eigen::Matrix3d R_cw = so3exp(w);
    Eigen::Vector3d y_wf = X(seqN(6,3));  

    Eigen::MatrixXd J_proj = proj_derivative(K_ * (R_cw * y_wf + t));

    J(Eigen::all, seqN(0,3)) = J_proj * K_;
    J(Eigen::all, seqN(3,3)) = J_proj * K_ * dR_wx_dw(w, y_wf);
    J(Eigen::all, seqN(6,3)) = J_proj * K_ * R_cw;    
    return J;
}
/********************************************************************************************/
/* Dynamics factor: constant-velocity model */
/*****************************************************************************************************/
DynamicsFactor::DynamicsFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
    float sigma, const Eigen::VectorXd& measurement, 
    float dt)
    : Factor{f_id, r_id, variables, sigma, measurement}{ 
        factor_type_ = DYNAMICS_FACTOR;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n_dofs_/2,n_dofs_/2);
        Eigen::MatrixXd O = Eigen::MatrixXd::Zero(n_dofs_/2,n_dofs_/2);
        Eigen::MatrixXd Qc_inv = pow(sigma, -2.) * I;

        Eigen::MatrixXd Qi_inv(n_dofs_, n_dofs_);
        Qi_inv << 12.*pow(dt, -3.) * Qc_inv,   -6.*pow(dt, -2.) * Qc_inv,
                  -6.*pow(dt, -2.) * Qc_inv,   4./dt * Qc_inv;   

        this->meas_model_lambda_ = Qi_inv;        

        // Store Jacobian as it is linear
        this->linear_ = true;
        J_ = Eigen::MatrixXd::Zero(n_dofs_, n_dofs_*2);
        J_ << I, dt*I, -1*I,    O,
             O,    I,    O, -1*I; 

    };

Eigen::MatrixXd DynamicsFactor::h_func_(const Eigen::VectorXd& X){
    return J_ * X;
}    
Eigen::MatrixXd DynamicsFactor::J_func_(const Eigen::VectorXd& X){
    return J_;
}

/********************************************************************************************/
/* Interrobot factor: for avoidance of other robots */
// This factor results in a high energy or cost if two robots are planning to be in the same 
// position at the same timestep (collision). This factor is created between variables of two robots.
// The factor has 0 energy if the variables are further away than the safety distance. skip_ = true in this case.
/********************************************************************************************/

InterrobotFactor::InterrobotFactor(int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
    float sigma, const Eigen::VectorXd& measurement, 
    float robot_radius)
    : Factor{f_id, r_id, variables, sigma, measurement} {  
        factor_type_ = INTERROBOT_FACTOR;
        float eps = 0.2 * robot_radius;
        this->safety_distance_ = 2*robot_radius + eps;
        this->delta_jac = 1e-2;
};

Eigen::MatrixXd InterrobotFactor::h_func_(const Eigen::VectorXd& X){
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(z_.rows(),z_.cols());
    Eigen::VectorXd X_diff = X(seqN(0,n_dofs_/2)) - X(seqN(n_dofs_, n_dofs_/2));
    X_diff += 1e-6*r_id_*Eigen::VectorXd::Ones(n_dofs_/2);

    double r = X_diff.norm();
    if (r <= safety_distance_){
        this->skip_flag = false;
        h(0) = 1.f*(1 - r/safety_distance_);
    }
    else {
        this->skip_flag = true;
    }

    return h;
};

Eigen::MatrixXd InterrobotFactor::J_func_(const Eigen::VectorXd& X){
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(z_.rows(), n_dofs_*2);
    Eigen::VectorXd X_diff = X(seqN(0,n_dofs_/2)) - X(seqN(n_dofs_, n_dofs_/2));
    X_diff += 1e-6*r_id_*Eigen::VectorXd::Ones(n_dofs_/2);// Add a tiny random offset to avoid div/0 errors
    double r = X_diff.norm();
    if (r <= safety_distance_){
        J(0,seqN(0, n_dofs_/2)) = -1.f/safety_distance_/r * X_diff;
        J(0,seqN(n_dofs_, n_dofs_/2)) = 1.f/safety_distance_/r * X_diff;
    }
    return J;
};

bool InterrobotFactor::skip_factor(){
    this->skip_flag = ( (X_(seqN(0,n_dofs_/2)) - X_(seqN(n_dofs_, n_dofs_/2))).squaredNorm() >= safety_distance_*safety_distance_ );
    return this->skip_flag;
}


/********************************************************************************************/
// Obstacle factor for static obstacles in the scene. This factor takes a pointer to the obstacle image from the Simulator.
// Note. in the obstacle image, white areas represent obstacles (as they have a value of 1).
// The input image to the simulator is opposite, which is why it needs to be inverted.
// The delta used in the first order jacobian calculation is chosen such that it represents one pixel in the image.
/********************************************************************************************/
ObstacleFactor::ObstacleFactor(Simulator* sim, int f_id, int r_id, std::vector<std::shared_ptr<Variable>> variables,
    float sigma, const Eigen::VectorXd& measurement, Image* p_obstacleImage)
    : Factor{f_id, r_id, variables, sigma, measurement}, p_obstacleImage_(p_obstacleImage){
        factor_type_ = OBSTACLE_FACTOR;
        this->delta_jac = 1.*(float)globals.WORLD_SZ / (float)p_obstacleImage->width;
};
Eigen::MatrixXd ObstacleFactor::h_func_(const Eigen::VectorXd& X){
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(1,1);
    // White areas are obstacles, so h(0) should return a 1 for these regions.
    float scale = p_obstacleImage_->width / (float)globals.WORLD_SZ;
    Vector3 c_hsv = ColorToHSV(GetImageColor(*p_obstacleImage_, (int)((X(0) + globals.WORLD_SZ/2) * scale), (int)((X(1) + globals.WORLD_SZ/2) * scale)));
    h(0) = c_hsv.z;
    return h;
};



///////////////////////////////////////////////////
// Lie Factors
///////////////////////////////////////////////////
FactorLie::FactorLie(int f_id, int r_id, std::vector<std::shared_ptr<VariableLie>> variables,
        float sigma, Eigen::VectorXd measurement, LieType measurement_lietype) 
        : f_id_(f_id), r_id_(r_id), key_(r_id, f_id), variables_(variables), z_(measurement), sigma_(sigma),
        measurement_lietype_(measurement_lietype) {

        // Initialise precision of the measurement function
        auto z_lt = lie_ndofs[measurement_lietype_];
        this->meas_model_lambda_ = Eigen::MatrixXd::Identity(z_lt, z_lt) / pow(sigma_, 2.);
        this->adaptive_gauss_noise_var_ = pow(sigma_, 2.);
        
        // Initialise empty inbox and outbox
        for (auto var : variables_) {
            inbox_[var->key_] = var->zero_msg_;
            outbox_[var->key_] = var->zero_msg_;
        }

        // This parameter useful if the factor is connected to another robot
        other_rid_=r_id_;                           

    };

/*****************************************************************************************************/
// Destructor
/*****************************************************************************************************/
FactorLie::~FactorLie(){};


bool FactorLie::update_factor(){

    // *Depending on the problem*, we may need to skip computation of this factor.
    // eg. to avoid extra computation, factor may not be required if two connected variables are too far apart.
    // in which case send out a Zero Message.
    if (this->skip_factor()){
        for (auto var : variables_){
            this->outbox_[var->key_] = var->zero_msg_;
        }           
        return false;
    }

    // Messages from connected variables are aggregated.
    // The beliefs are used to create the linearisation point X_.
    int idx = 0;
    lin_point_.clear();
    for (int v=0; v<variables_.size(); v++){
        auto& [X_in, Lambda_in, _] = this->inbox_[variables_[v]->key_];
        lin_point_.push_back(X_in);
        idx += lie_ndofs[lietypes_[v]];
    }
    
    auto [residual, J] = computeResidualJacobian();
    this->initialised_ = true;

    Eigen::MatrixXd factor_lam_potential;
    Eigen::VectorXd factor_eta_potential;
    factor_lam_potential = J.transpose() * meas_model_lambda_ * J;
    factor_eta_potential = J.transpose() * meas_model_lambda_ * residual;

    /**** Robustify with HUBER loss *****/
    double robustness_k = 1.;
    double mahalanobis_dist = residual.norm() / this->sigma_;
    if (mahalanobis_dist >= mahalanobis_threshold_){
        robustness_k = (2. * mahalanobis_threshold_ * mahalanobis_dist - pow(mahalanobis_threshold_, 2.)) / pow(mahalanobis_dist, 2.);
    }

    factor_eta_potential *= robustness_k;
    factor_lam_potential *= robustness_k;


    //  Update factor precision and information with incoming messages from connected variables.
    int marginalisation_idx = 0;
    for (int v_out_idx=0; v_out_idx<variables_.size(); v_out_idx++){
        auto var_out = variables_[v_out_idx];
        // Initialise with factor values
        Eigen::VectorXd factor_eta = factor_eta_potential;     
        Eigen::MatrixXd factor_lam = factor_lam_potential;
        
        // Combine the factor with the belief from other variables apart from the receiving variable
        int idx_v = 0;
        for (int v_idx=0; v_idx<variables_.size(); v_idx++){
            int n_dofs = variables_[v_idx]->n_dofs_;
            if (variables_[v_idx]->key_ != var_out->key_) {
                auto lt = lietypes_[v_idx];
                auto [Xin, Lin, _] = inbox_[variables_[v_idx]->key_];
                auto tau = rightminus(Xin, lin_point_[v_idx], lt);
                Eigen::MatrixXd Lam = rjac(tau, lt).transpose() * Lin * rjac(tau, lt);
                Eigen::VectorXd eta = Lam * tau;     
                factor_eta(seqN(idx_v, n_dofs)) += eta;
                factor_lam(seqN(idx_v, n_dofs), seqN(idx_v, n_dofs)) += Lam;
            }
            idx_v += n_dofs;
        }
        
        // Marginalise the Factor Precision and Information to send to the relevant variable
        auto [eta_marg, Lam_marg] = marginalise_factor_dist(factor_eta, factor_lam, v_out_idx, marginalisation_idx);

        // Apply damping and send out
        // outgoing_msg.eta *= (1. - damping_);
        // outgoing_msg.lambda *= (1. - damping_);

        auto lt = lietypes_[v_out_idx];
        Eigen::VectorXd tau = Lam_marg.colPivHouseholderQr().solve(eta_marg);
        auto X_out = rightplus(lin_point_[v_out_idx], tau, lt);

        Eigen::MatrixXd L_out = rjacinv(tau, lt).transpose() * Lam_marg * rjacinv(tau, lt);
        outbox_[var_out->key_] = MessageLie(X_out, L_out);

        marginalisation_idx += var_out->n_dofs_;
    }

    return true;
};

/*****************************************************************************************************/
// Marginalise the factor Precision and Information and create the outgoing message to the variable
/*****************************************************************************************************/
std::pair<Eigen::VectorXd, Eigen::MatrixXd> FactorLie::marginalise_factor_dist(const Eigen::VectorXd &eta, const Eigen::MatrixXd &Lam, int var_idx, int marg_idx){
    // Marginalisation only needed if factor is connected to >1 variables
    int n_dofs = variables_[var_idx]->n_dofs_;
    if (eta.size() == n_dofs) return {eta, Lam};

    Eigen::VectorXd eta_a(n_dofs), eta_b(eta.size()-n_dofs);
    eta_a = eta(seqN(marg_idx, n_dofs));
    eta_b << eta(seq(0, marg_idx - 1)), eta(seq(marg_idx + n_dofs, last));

    Eigen::MatrixXd lam_aa(n_dofs, n_dofs), lam_ab(n_dofs, Lam.cols()-n_dofs);
    Eigen::MatrixXd lam_ba(Lam.rows()-n_dofs, n_dofs), lam_bb(Lam.rows()-n_dofs, Lam.cols()-n_dofs);
    lam_aa << Lam(seqN(marg_idx, n_dofs), seqN(marg_idx, n_dofs));
    lam_ab << Lam(seqN(marg_idx, n_dofs), seq(0, marg_idx - 1)), Lam(seqN(marg_idx, n_dofs), seq(marg_idx + n_dofs, last));
    lam_ba << Lam(seq(0, marg_idx - 1), seq(marg_idx, marg_idx + n_dofs - 1)), Lam(seq(marg_idx + n_dofs, last), seqN(marg_idx, n_dofs));
    lam_bb << Lam(seq(0, marg_idx - 1), seq(0, marg_idx - 1)), Lam(seq(0, marg_idx - 1), seq(marg_idx + n_dofs, last)),
            Lam(seq(marg_idx + n_dofs, last), seq(0, marg_idx - 1)), Lam(seq(marg_idx + n_dofs, last), seq(marg_idx + n_dofs, last));

    Eigen::MatrixXd lam_bb_inv = lam_bb.inverse();
    Message marginalised_msg(n_dofs);
    Eigen::VectorXd eta_marg = eta_a - lam_ab * lam_bb_inv * eta_b;
    Eigen::MatrixXd lambda_marg = lam_aa - lam_ab * lam_bb_inv * lam_ba;
    if (!lambda_marg.allFinite()) {
        lambda_marg = lambda_marg * 0.;
        eta_marg = eta_marg * 0.;
    }

    return {eta_marg, lambda_marg};
};    

/********************************************************************************************/
/* PriorFactor factor: */
/*****************************************************************************************************/
PriorFactor::PriorFactor(int f_id, int r_id, std::vector<std::shared_ptr<VariableLie>> variables,
            float sigma, Eigen::VectorXd measurement, LieType lietype)
    : FactorLie{f_id, r_id, variables, sigma, measurement, lietype}{ 
        factor_type_ = DEFAULT_FACTOR;
        lietypes_ = std::vector<LieType>{lietype};
        n_dofs_ = lie_ndofs[lietype];
    };

std::pair<Eigen::VectorXd, Eigen::MatrixXd> PriorFactor::computeResidualJacobian(){
    Eigen::VectorXd h;    
    Eigen::MatrixXd J_h_X(n_dofs_, n_dofs_); 
    h = rightminus(lin_point_[0], z_, lietypes_[0], J_h_X);    // h is in tangent space
    Eigen::VectorXd res = -h; // [0] - h
    return {res, J_h_X};
};   
/********************************************************************************************/
/* Smoothness factor: */
/*****************************************************************************************************/
SmoothnessFactor::SmoothnessFactor(int f_id, int r_id, std::vector<std::shared_ptr<VariableLie>> variables,
            float sigma, Eigen::VectorXd measurement, LieType lietype)
    : FactorLie{f_id, r_id, variables, sigma, measurement, lietype}{ 
        factor_type_ = DEFAULT_FACTOR;
        lietypes_ = std::vector<LieType>{lietype, lietype};
        n_dofs_ = lie_ndofs[lietype];
    };

std::pair<Eigen::VectorXd, Eigen::MatrixXd> SmoothnessFactor::computeResidualJacobian(){
    Eigen::VectorXd h;    
    Eigen::MatrixXd J(n_dofs_,n_dofs_*variables_.size());
    auto& X1 = lin_point_[0], X2 = lin_point_[1];
    Eigen::Matrix3d J_x1mx2_x1, J_x1mx2_x2, J_h_exp, J_exp_Jx1mx2;
    
    // h = Exp( x1 (-) x2 ) (-) z
    LieType LT = lietypes_[0];
    h = rightminus( Exp( rightminus( X1, X2, LT, J_x1mx2_x1, J_x1mx2_x2), LT, J_exp_Jx1mx2), z_, LT, J_h_exp);
    J << (J_h_exp * J_exp_Jx1mx2 * J_x1mx2_x1).eval(), (J_h_exp * J_exp_Jx1mx2 * J_x1mx2_x2).eval();
    Eigen::VectorXd res = -h; // [0] - h
    return {res, J};
};
/********************************************************************************************/
/* AngleDifferenceFactorSE2d factor: */
/*****************************************************************************************************/
AngleDifferenceFactorSE2d::AngleDifferenceFactorSE2d(int f_id, int r_id, std::vector<std::shared_ptr<VariableLie>> variables,
            float sigma, Eigen::VectorXd measurement)
    : FactorLie{f_id, r_id, variables, sigma, measurement, LieType::SE2d}{ 
        factor_type_ = DEFAULT_FACTOR;
        lietypes_ = std::vector<LieType>{LieType::SE2d, LieType::SE2d};
    };

std::pair<Eigen::VectorXd, Eigen::MatrixXd> AngleDifferenceFactorSE2d::computeResidualJacobian(){
    Eigen::VectorXd h;    
    Eigen::MatrixXd J;
    // Measurement function h(X1,X2) = theta1 - theta2
    auto& X1 = lin_point_[0], X2 = lin_point_[1];
    J = Eigen::MatrixXd::Zero(3,6);
    J << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity();
    J(2,5) = -1;
    double angular_diff = manif::SE2d(X1).angle() - manif::SE2d(X2).angle();
    h = manif::SE2d(0., 0., angular_diff).coeffs();
    return {rightminus(z_, h, LieType::SE2d), J};
}   

/********************************************************************************************/
/* Dynamics factor: constant-velocity model */
/*****************************************************************************************************/
DynamicsFactorSE2_2d::DynamicsFactorSE2_2d(int f_id, int r_id, std::vector<std::shared_ptr<VariableLie>> variables,
    float sigma, const Eigen::VectorXd& measurement, float dt)
    : dt_(dt), FactorLie{f_id, r_id, variables, sigma, measurement, LieType::SE2d}{ 
        factor_type_ = DEFAULT_FACTOR;
        lietypes_ = std::vector<LieType>{LieType::SE2_2d, LieType::SE2_2d};

        n_dofs_ = lie_ndofs[LieType::SE2_2d];
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n_dofs_/2,n_dofs_/2);
        Eigen::MatrixXd O = Eigen::MatrixXd::Zero(n_dofs_/2,n_dofs_/2);
        Eigen::MatrixXd Qc_inv = pow(sigma, -2.) * I;

        Eigen::MatrixXd Qi_inv(n_dofs_, n_dofs_);
        Qi_inv << 12.*pow(dt, -3.) * Qc_inv,   -6.*pow(dt, -2.) * Qc_inv,
                  -6.*pow(dt, -2.) * Qc_inv,   4./dt * Qc_inv;   

        this->meas_model_lambda_ = Qi_inv;        

        // Store Jacobian as it is linear
        // this->linear_ = true;
        // J_ = Eigen::MatrixXd::Zero(n_dofs_, n_dofs_*2);
        // J_ << I, dt*I, -1*I,    O,
            //  O,    I,    O, -1*I; 

    };

std::pair<Eigen::VectorXd, Eigen::MatrixXd> DynamicsFactorSE2_2d::computeResidualJacobian(){
    Eigen::VectorXd h(n_dofs_);    
    Eigen::MatrixXd J(n_dofs_, 2*n_dofs_);
    // Measurement function h(X1,X2) = X1 + dt*X1_dot - X2
    int n = lin_point_[0].size();
    Eigen::VectorXd X1 = lin_point_[0](Eigen::seqN(0, n/2)), X2 = lin_point_[1](Eigen::seqN(0, n/2));
    Eigen::VectorXd X1_dot = lin_point_[0](Eigen::seqN(n/2, n/2)), X2_dot = lin_point_[1](Eigen::seqN(n/2, n/2));

    Eigen::Matrix3d J_logx1dot_x1dot;
    Eigen::Matrix3d J_res_h0, J_exp_x1comb, J_x1comb_x1, J_h0_x2, J_x1comb_logx1dot, J_h0_exp;
    Eigen::Matrix3d J_exp_x1dot, J_exp_x2dot, J_res_h1, J_h1_exp;
    // res = h(x1comb(x1, Log(x1dot)) - x2) - z
    h(Eigen::seqN(0, 3)) = rightminus(  
                Exp(                                                         // res
                    rightminus(
                        rightplus(X1, 
                            dt_ * Log(X1_dot, 
                                LieType::SE2d, J_logx1dot_x1dot), 
                            LieType::SE2d, J_x1comb_x1, J_x1comb_logx1dot),
                    X2, LieType::SE2d, J_exp_x1comb, J_h0_x2), 
                LieType::SE2d, J_h0_exp),
            z_({0,1,2,3}), LieType::SE2d, J_res_h0);

    h(Eigen::seqN(3, 3)) = rightminus(
                Exp(
                    rightminus(
                        X1_dot, X2_dot, LieType::SE2d,
                        J_exp_x1dot, J_exp_x2dot
                    ), LieType::SE2d, J_h1_exp),
                z_({4,5,6,7}), LieType::SE2d, J_res_h1);

    J(Eigen::seqN(0, n_dofs_/2), Eigen::seqN(0, 3)) = (J_res_h0 * J_h0_exp * J_exp_x1comb * J_x1comb_x1).eval();
    J(Eigen::seqN(0, n_dofs_/2), Eigen::seqN(3, 3)) = (J_res_h0 * J_h0_exp * J_exp_x1comb * J_x1comb_logx1dot * dt_ * J_logx1dot_x1dot).eval();
    J(Eigen::seqN(0, n_dofs_/2), Eigen::seqN(6, 3)) = (J_res_h0 * J_h0_x2).eval();
    J(Eigen::seqN(0, n_dofs_/2), Eigen::seqN(9, 3)) = Eigen::Matrix3d::Zero();
    
    J(Eigen::seqN(3, n_dofs_/2), Eigen::seqN(0, 3)) = Eigen::Matrix3d::Zero();
    J(Eigen::seqN(3, n_dofs_/2), Eigen::seqN(3, 3)) = J_res_h1 * J_h1_exp * J_exp_x1dot;
    J(Eigen::seqN(3, n_dofs_/2), Eigen::seqN(6, 3)) = Eigen::Matrix3d::Zero();
    J(Eigen::seqN(3, n_dofs_/2), Eigen::seqN(9, 3)) = J_res_h1 * J_h1_exp * J_exp_x2dot;

    Eigen::VectorXd res = -h; // [0] - h
    return {res, J};            
}   