/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed under MIT license (see LICENSE for details)
/**************************************************************************************/
#include <gbp/Factorgraph.h>
#include <gbp/GBPCore.h>

/******************************************************************************************************/
// This is for the FactorGraph class.
// The robot class uses the FactorGraph class.
/******************************************************************************************************/
FactorGraph::FactorGraph(int robot_id) : robot_id_(robot_id){};

/******************************************************************************************************/
// Factor Iteration in Gaussian Belief Propagation (GBP).
// For each factor in the factorgraph:
//  - Messages are collected from the outboxes of each of the connected variables
//  - Factor potential is calculated and outgoing message in the factor's outbox is created.
//
//  * Note: we deal with cases where the variable/factor iteration may need to be skipped:
//      - communications failure modes:
//          if interrobot_comms_active_ is false, variables and factors connected to 
//          other robots should not take part in GBP iterations,
//      - message passing modes (INTERNAL within a robot's own factorgraph or EXTERNAL between a robot and other robots):
//          in which case the variable or factor may or may not need to take part in GBP depending on if it's connected to another robot
/******************************************************************************************************/
void FactorGraph::factorIteration(MsgPassingMode msg_passing_mode){
// #pragma omp parallel for    
    for (int f_idx=0; f_idx<active_factors_.size(); f_idx++){
        auto f_it = active_factors_.begin(); std::advance(f_it, f_idx);
        auto fac = factors_.at(Key{0, *f_it});
        auto f_key = fac->key_;

        // if (!fac->active_) continue;

        for (auto var : fac->variables_){
            if (!var->active_) continue;
            // Read message from each connected variable
            fac->inbox_[var->key_] = var->outbox_.at(f_key);
        }
        // Calculate factor potential and create outgoing messages
        fac->update_factor();
    };
};

/******************************************************************************************************/
// Variable Iteration in Gaussian Belief Propagation (GBP).
// For each variable in the factorgraph:
//  - Messages are collected from the outboxes of each of the connected factors
//  - Variable belief is updated and outgoing message in the variable's outbox is created.
//
//  * Note: we deal with cases where the variable/factor iteration may need to be skipped:
//      - communications failure modes:
//          if interrobot_comms_active_ is false, variables and factors connected to 
//          other robots should not take part in GBP iterations,
//      - message passing modes (INTERNAL within a robot's own factorgraph or EXTERNAL between a robot and other robots):
//          in which case the variable or factor may or may not need to take part in GBP depending on if it's connected to another robot
/******************************************************************************************************/
void FactorGraph::variableIteration(MsgPassingMode msg_passing_mode){
// #pragma omp parallel for    
    for (int v_idx=0; v_idx<active_variables_.size(); v_idx++){
        auto v_it = active_variables_.begin(); std::advance(v_it, v_idx);
        auto var = variables_.at(Key{0, *v_it});
        auto v_key = var->key_;
        
        // if (!var->active_) continue;

        for (auto [f_key, fac] : var->factors_){
            if (!fac->active_) continue;
            // Read message from each connected factor
            var->inbox_[f_key] = fac->outbox_.at(v_key);
        }

        // Update variable belief and create outgoing messages
        var->update_belief();
    };
;}
