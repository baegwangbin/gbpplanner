/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed under MIT license (see LICENSE for details)
/**************************************************************************************/
#pragma once

#include <vector>
#include <math.h>
#include <memory>

#include <Utils.h>
#include <gbp/GBPCore.h>
#include <manif/SE2.h>
#include <manif/SE3.h>
#include <manif/SO2.h>
#include <manif/SO3.h>

#include <raylib.h>
#include <Eigen/Core>
#include <Eigen/Dense>

class Factor;    // Forward declaration
class FactorLie;    // Forward declaration
/***********************************************************************************************************/
// Variable used in GBP
/***********************************************************************************************************/
class Variable {
    public:
        int v_id_;                                          // id of variable
        int r_id_;                                          // id of robot this variable belongs to
        Key key_;                                           // Key {r_id_, v_id_}
        Eigen::VectorXd eta_prior_;                         // Information vector of prior on variable (essentially like a unary factor)
        Eigen::MatrixXd lam_prior_;                         // Precision matrix of prior on variable (essentially like a unary factor)
        int n_dofs_;                                        // Degrees of freedom of variable. For 2D case n_dofs_ = 4 ([x,y,xdot,ydot])
        Message belief_;                                    // Belief of variable
        Eigen::VectorXd eta_;                               // Information vector of variable's belief
        Eigen::MatrixXd lam_;                               // Precision matrix of variable's belief
        Eigen::VectorXd mu_;                                // Mean vector of variable's belief
        Eigen::MatrixXd sigma_;                             // sqrt(covariance) of variable's belief
        Mailbox inbox_, outbox_;                            // Mailboxes for message storage                    
        bool valid_ = false;                                // Flag whether variable's covariance is finite
        std::map<Key, std::shared_ptr<Factor>> factors_{};  // Map of factors connected to the variable, accessed by their key
        float size_;                                        // Size of variable (usually taken from robot->robot_radius)
        std::function<void()> draw_fn_ = NULL;              // Space for custom draw function of variable. Usually robot->draw() supercedes this
        bool active_ = true;

        // Function declarations
        Variable(int v_id, int r_id, const Eigen::VectorXd& mu_prior, const Eigen::VectorXd& sigma_prior_list, float size, int n_dofs=4);
        
        ~Variable();

        void update_belief();
        
        void change_variable_prior(const Eigen::VectorXd& new_mu);

        void add_factor(std::shared_ptr<Factor> fac);

        void delete_factor(Key fac_key);

        virtual void draw(bool filled=true);

};

class VariableLie {
    public:
        LieType lietype_;
        int v_id_;                                          // id of variable
        int r_id_;                                          // id of robot this variable belongs to
        Key key_;                                           // Key {r_id_, v_id_}
        Eigen::VectorXd state_;
        int n_dofs_;                                        // Degrees of freedom of variable. For 2D case n_dofs_ = 4 ([x,y,xdot,ydot])
        Eigen::VectorXd eta_;
        Eigen::MatrixXd lam_;
        Eigen::MatrixXd sigma_;                             // sqrt(covariance) of variable's belief
        MailboxLie inbox_, outbox_;                            // Mailboxes for message storage   
        MessageLie zero_msg_;
        bool valid_ = false;                                // Flag whether variable's covariance is finite
        std::map<Key, std::shared_ptr<FactorLie>> factors_{};  // Map of factors connected to the variable, accessed by their key
        float size_;                                        // Size of variable (usually taken from robot->robot_radius)
        std::function<void()> draw_fn_ = NULL;              // Space for custom draw function of variable. Usually robot->draw() supercedes this
        std::shared_ptr<FactorLie> prior_factor_;
        

        // Function declarations
        VariableLie(LieType lietype, int v_id, int r_id, const Eigen::VectorXd& sigma_prior_list);

        // ~VariableLie();

        void update_belief();
        
        // void change_variable_prior(const Eigen::VectorXd& new_mu);

        void add_factor(std::shared_ptr<FactorLie> fac, bool is_prior=false);

        // void delete_factor(Key fac_key);

        // virtual void draw(bool filled=true);

};
