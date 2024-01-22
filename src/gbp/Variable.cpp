/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed under MIT license (see LICENSE for details)
/**************************************************************************************/
#include <Eigen/Core>
#include <Eigen/Dense>
#include <raylib.h>
#include <vector>
#include <math.h>
#include <Utils.h>
#include <gbp/GBPCore.h>

#include <gbp/Variable.h>
#include <gbp/Factor.h>

/***********************************************************************************************************/
// Variable constructor
// Inputs:
// - variable id (vid, taken from simulator->next_v_id_++)
// - id of the robot this variable belongs to
// - mu_prior: mean vector (mu) of the prior on the variable
// - sigma_prior_list: an Eigen::VectorXd of length n_dofs representing the strength of the prior
//      (eg. sigma_prior_list(0)^2 is the covariance of the first element of the variable (x coordinate).
//      This is used to create the Precision (Lambda) of the prior as Lambda = diag(sigma_prior_list)
/***********************************************************************************************************/
Variable::Variable(int v_id, int r_id, const Eigen::VectorXd& mu_prior, const Eigen::VectorXd& sigma_prior_list,
                   float size, int n_dofs)
            : v_id_(v_id), r_id_(r_id), key_(Key(r_id, v_id)), mu_(mu_prior), size_(size), n_dofs_(n_dofs) {
                lam_prior_ = sigma_prior_list.cwiseProduct(sigma_prior_list).cwiseInverse().asDiagonal();
                if (!lam_prior_.allFinite()) lam_prior_.setZero();
                eta_prior_ = lam_prior_ * mu_prior;
                belief_ = Message(eta_prior_, lam_prior_, mu_prior);

            };

/***********************************************************************************************************/
// Destructor: deletes all connected factors too
/***********************************************************************************************************/
Variable::~Variable(){
    for (auto [fkey, fac] : factors_){
        delete_factor(fkey);
    }
}

/***********************************************************************************************************/
// Variable belief update step:
// Aggregates all the messages from its connected factors (begins with the prior, as this is effectively a unary factor)
// The valid_ flag is useful for drawing the variable.
// Finally the outgoing messages to factors is created.
/***********************************************************************************************************/
void Variable::update_belief(){
    // Collect messages from all other factors, begin by "collecting message from pose factor prior"
    eta_ = eta_prior_;
    lam_ = lam_prior_;

    for (auto& [f_key, msg] : inbox_) {
        auto [eta_msg, lam_msg, _] = msg;
        eta_ += eta_msg;
        lam_ += lam_msg;
    }
    // Update belief
    sigma_ = lam_.inverse();
    valid_ = sigma_.allFinite();
    if (valid_) mu_ = sigma_ * eta_;
    belief_ = Message {eta_, lam_, mu_};

    // Create message to send to each factor that sent it stuff
    // msg is the aggregate of all OTHER factor messages (belief - last sent msg of that factor)
    for (auto [f_key, fac] : factors_) {
        outbox_[f_key] = belief_ - inbox_.at(f_key);
    }
}

/***********************************************************************************************************/
// Changes the prior on the variable. It updates the belief_ of the variable.
/***********************************************************************************************************/
void Variable::change_variable_prior(const Eigen::VectorXd& new_mu){
    eta_prior_ = lam_prior_ * new_mu;
    mu_ = new_mu;
    belief_ = Message {eta_, lam_, mu_};
    for (auto [fkey, fac] : factors_){
        outbox_[fkey] = belief_;
        inbox_[fkey].setZero();
    }
};

/***********************************************************************************************************/
// Add a factor to this variable's list of factors, and initialise an outgoing message of its belief.
/***********************************************************************************************************/
void Variable::add_factor(std::shared_ptr<Factor> fac){
    factors_[fac->key_] = fac;
    inbox_[fac->key_] = Message(n_dofs_);
    outbox_[fac->key_] = belief_;
}

/***********************************************************************************************************/
// Delete a factor from this variable's list of factors. Remove it from its inbox too.
/***********************************************************************************************************/
void Variable::delete_factor(Key fac_key){
    factors_.erase(fac_key);
    inbox_.erase(fac_key);
}

/***********************************************************************************************************/
// Default drawing function for the variable. A custom function can be set by setting draw_fn_ = a lambda[](){} function
/***********************************************************************************************************/
void Variable::draw(bool filled){
    if (draw_fn_!=NULL){
        draw_fn_;
    } else {
        // Default variable draw function.
        if (!valid_) return;
        const int var_x = (mu_(0) );
        const int var_y = (mu_(1) );
        DrawSphere(Vector3{(float)var_x, size_, (float)var_y}, size_, BLACK);
    }
}


VariableLie::VariableLie(LieType lietype, int v_id, int r_id, const Eigen::VectorXd& sigma_prior_list) :
                            lietype_(lietype), v_id_(v_id), r_id_(r_id), key_(Key(r_id, v_id)) {
        n_dofs_ = lie_ndofs[lietype_];
        state_ = lie_identity_coeffs(lietype_);
        eta_ = Eigen::VectorXd::Zero(n_dofs_);                               // Information vector of variable's belief
        lam_ = Eigen::MatrixXd::Zero(n_dofs_, n_dofs_);                            
        zero_msg_ = MessageLie(n_dofs_);                                
};

void VariableLie::update_belief(){
    // Collect messages from all other factors, begin by "collecting message from pose factor prior"
    eta_.setZero();
    lam_.setZero();

    for (auto& [f_key, msg] : inbox_) {
        auto [X_in, lam_msg, _] = msg;
        Eigen::VectorXd tau = rightminus(X_in, state_, lietype_);
        Eigen::MatrixXd Lambda = rjac(tau, lietype_).transpose() * lam_msg * rjac(tau, lietype_);
        eta_ += Lambda * tau;
        lam_ += Lambda;
    }

    // Create message to send to each factor that sent it stuff
    // msg is the aggregate of all OTHER factor messages (belief - last sent msg of that factor)
    for (auto [f_key, fac] : factors_) {
        auto [X_in, lam_msg, _] = inbox_.at(f_key);
        Eigen::VectorXd tau = rightminus(X_in, state_, lietype_);
        Eigen::MatrixXd L_msg_in = rjac(tau, lietype_).transpose() * lam_msg * rjac(tau, lietype_);
        Eigen::VectorXd eta_out = eta_ - L_msg_in * tau;
        Eigen::MatrixXd L_out = lam_ - L_msg_in;
        Eigen::VectorXd tau_out = L_out.colPivHouseholderQr().solve(eta_out); 
        outbox_[f_key] = MessageLie(rightplus(state_, tau_out, lietype_), rjacinv(tau_out, lietype_).transpose() * L_out * rjacinv(tau_out, lietype_));
    }

    Eigen::VectorXd tau = lam_.colPivHouseholderQr().solve(eta_); 
    this->state_ = rightplus(state_, tau, lietype_);
}

void VariableLie::add_factor(std::shared_ptr<FactorLie> fac, bool is_prior){
    if (is_prior) prior_factor_ = fac;
    factors_[fac->key_] = fac;
    inbox_[fac->key_] = MessageLie(lie_ndofs[lietype_]);
    outbox_[fac->key_] = MessageLie(state_, lam_);
}