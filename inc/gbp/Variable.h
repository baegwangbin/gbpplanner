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
#include <typeindex>
#include <typeinfo>

class Factor;    // Forward declaration
template <class MeasType, class VarsType>
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

class VariableLieBase {
    public:
        int v_id_=5;
        int r_id_;
        Key key_ = Key(-1, -1);
        int n_dofs_= -1;
        void* temp_ptr;
        MailboxLieVariable<manif::SE2d> inbox_, outbox_;                            // Mailboxes for message storage                    
        size_t typecode_;

        VariableLieBase(){};
        // virtual void add_factor(std::shared_ptr<FactorLie<manif::SE2d, manif::SE2d>> fac) = 0;
        // virtual void add_factor(std::shared_ptr<FactorLie<manif::SO3d, manif::SO3d>> fac) = 0;
        virtual void init() = 0;
        virtual void temp() = 0;

};

template<class LieGroupType>
class VariableLie: virtual public VariableLieBase {
    public:
        int v_id_;                                          // id of variable
        int r_id_;                                          // id of robot this variable belongs to
        Key key_;                                           // Key {r_id_, v_id_}
        LieGroupType state_ = LieGroupType::Identity();
        using LieGroupTangentType = typename LieGroupType::Tangent;
        int n_dofs_ = state_.DoF;                                        // Degrees of freedom of variable. For 2D case n_dofs_ = 4 ([x,y,xdot,ydot])
        Eigen::VectorXd eta_ = Eigen::VectorXd::Zero(n_dofs_);                               // Information vector of variable's belief
        Eigen::MatrixXd lam_ = Eigen::MatrixXd::Zero(n_dofs_, n_dofs_);                               // Precision matrix of variable's belief
        Eigen::MatrixXd sigma_;                             // sqrt(covariance) of variable's belief
        MailboxLieVariable<LieGroupType> inbox_, outbox_;                            // Mailboxes for message storage   
        MessageLie<LieGroupType> zero_msg_ = MessageLie<LieGroupType>();                 
        bool valid_ = false;                                // Flag whether variable's covariance is finite
        std::map<Key, std::shared_ptr<FactorLie<LieGroupType, LieGroupType>>> factors_{};  // Map of factors connected to the variable, accessed by their key
        float size_;                                        // Size of variable (usually taken from robot->robot_radius)
        std::function<void()> draw_fn_ = NULL;              // Space for custom draw function of variable. Usually robot->draw() supercedes this
        bool active_ = true;

        // Function declarations
        VariableLie(int v_id, int r_id, const Eigen::VectorXd& sigma_prior_list);
        
        // ~VariableLie();
        void init();

        void update_belief();
        void temp();
        
        // void change_variable_prior(const Eigen::VectorXd& new_mu);

        void add_factor(std::shared_ptr<FactorLie<LieGroupType, LieGroupType>> fac);

        // void delete_factor(Key fac_key);

        // virtual void draw(bool filled=true);

};