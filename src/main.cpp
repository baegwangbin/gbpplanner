/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed under MIT license (see LICENSE for details)

// Define all parameters in the appropriate config file (default: config/config.json)
/**************************************************************************************/
#define RLIGHTS_IMPLEMENTATION // needed to be defined once for the lights shader
#include <iostream>
#include <Utils.h>

#include <DArgs.h>

#include "definitions.h"
#include <Globals.h>
#include <gbp/GBPCore.h>
#include <gbp/Factorgraph.h>
#include <Simulator.h>
#include <Eigen/Geometry>
#include <map>
#include <gbp/lie_algebra.h>
#include <any>
#include <tuple>
#include <variant>
#include <typeinfo>
#include <typeindex>
#include <manif/manif.h>
#include "manif/Bundle.h"
Globals globals;
Graphics* graphics;
std::map<int, std::shared_ptr<FactorGraph>> factorgraphs{};
std::shared_ptr<FactorGraph> factorgraph;
void loadFrame();
void eventHandler();
void draw();
void drawCameraVar(std::shared_ptr<Variable> cam_var);
void drawLandmarkVar(std::shared_ptr<Variable> lmk_var);
void drawWorldFrame();
int next_cam_id = 0;
int next_vid_ = 0;
int next_fid_ = 0;
int LMK_ID_OFFSET = 0;


int n_dof_lmk = 3;
int n_dof_cam = 6;
double sigma_lmk = 1;
double sigma_cam = 1.;
double sigma_reproj = 2.;
std::vector<Eigen::VectorXd> cam_means;
std::vector<Eigen::VectorXd> lmk_means;
std::map<int, std::map<int, Eigen::VectorXd>> meas_dict;
std::map<int, std::shared_ptr<Variable>> inactive_vars{};
std::map<int, int> lmk2vid{}; // lmk_id, num
std::map<int, int> cam2vid{}; // cam_id, num
int cam_iter = 0;

Eigen::MatrixXd K;

Model cframe;

int next_vid = 0;
int next_fid = 0;
double sigma_prior = 10.;
double sigma_smoothness = 0.001;
std::vector<std::shared_ptr<VariableLie>> all_variables{};
std::vector<std::shared_ptr<FactorLie>> all_factors{};


int main(int argc, char *argv[]){

    srand((int)globals.SEED);                                   // Initialise random seed   
    DArgs::DArgs dargs(argc, argv);                             // Parse config file argument --cfg <file.json>
    if (globals.parse_global_args(dargs)) return EXIT_FAILURE;
    SetTraceLogLevel(LOG_ERROR);  
    SetTargetFPS(60);
    InitWindow(globals.SCREEN_SZ, globals.SCREEN_SZ, globals.WINDOW_TITLE);
    Image temp_img = GenImageColor(500, 500, WHITE);
    graphics = new Graphics(temp_img);
    globals.SIM_MODE = SimNone;

    /////////////////////////////////////////////////////////////////
    // Create variables
    int num_intermediate_vars = 10;
    int v1_id = next_vid++;
    int _rid = 1;
    Eigen::VectorXd siglist{{1., 1., 1.}};
    // auto v1 = std::make_shared<VariableLie>(LieType::SO2d, v1_id, _rid, siglist);
    auto v1 = std::make_shared<VariableLie>(LieType::SE2d, v1_id, _rid, siglist);
    all_variables.push_back(v1);

    // Add prior on pose
    int v1_prior_id = next_fid++;
    auto v1_prior_state = manif::SE2d(0.,0., 0*DEG2RAD).coeffs();
    auto f1 = std::make_shared<PriorFactor>(v1_prior_id, _rid, std::vector<std::shared_ptr<VariableLie>>{v1}, 1e-10*sigma_prior, v1_prior_state, LieType::SE2d);
    // auto v1_prior_state = manif::SO2d(350*DEG2RAD).coeffs();
    // auto f1 = std::make_shared<PriorFactor>(v1_prior_id, _rid, std::vector<std::shared_ptr<VariableLie>>{v1}, 0.001*1.*sigma_prior, v1_prior_state, LieType::SO2d);
    all_factors.push_back(f1);
    v1->add_factor(f1, true);

    for (int i=0; i<num_intermediate_vars; i++){
        int vid = next_vid++;
        auto var = std::make_shared<VariableLie>(LieType::SE2d, vid, _rid, siglist);
        all_variables.push_back(var);

        // Add prior on pose
        int v_p_id = next_fid++;
        auto v_prior_state = manif::SE2d(0., 0., 0.*DEG2RAD).coeffs();
        auto f2 = std::make_shared<PriorFactor>(v_p_id, _rid, std::vector<std::shared_ptr<VariableLie>>{var}, 10.*sigma_prior, v_prior_state, LieType::SE2d);
        all_factors.push_back(f2);
        var->add_factor(f2, true);
    }

    int v2_id = next_vid++;
    // auto v2 = std::make_shared<VariableLie>(LieType::SO2d,v2_id, _rid, siglist);
    auto v2 = std::make_shared<VariableLie>(LieType::SE2d,v2_id, _rid, siglist);
    all_variables.push_back(v2);

    // Add prior on pose
    int v2_prior_id = next_fid++;
    auto v2_prior_state = manif::SE2d(4., 0., 180*DEG2RAD).coeffs();
    auto f2 = std::make_shared<PriorFactor>(v2_prior_id, _rid, std::vector<std::shared_ptr<VariableLie>>{v2}, 1e-10*sigma_prior, v2_prior_state, LieType::SE2d);
    // auto v2_prior_state = manif::SO2d(10*DEG2RAD).coeffs();
    // auto f2 = std::make_shared<PriorFactor>(v2_prior_id, _rid, std::vector<std::shared_ptr<VariableLie>>{v2}, 1.*sigma_prior, v2_prior_state, LieType::SO2d);
    all_factors.push_back(f2);
    v2->add_factor(f2, true);
    // /////////////////////////////////////////////////////////////////
    
    // int f12_id = next_fid++;
    // std::vector<std::shared_ptr<VariableLie>> _variables {v1, v2};
    // auto measurement = manif::SE2d(0., 0., PI/4.).coeffs();
    // auto f12 = std::make_shared<AngleDifferenceFactorSE2d>(f12_id, 0, _variables, sigma_smoothness, measurement);
    // all_factors.push_back(f12);
    // for (auto v : _variables){
    //     v->add_factor(f12);
    // };
    for (int ii=0; ii<num_intermediate_vars+1; ii++){
        int f12_id = next_fid++;
        std::vector<std::shared_ptr<VariableLie>> variables {all_variables[ii], all_variables[ii+1]};
        auto measurement = manif::SE2d(0.,0.,0.).coeffs();
        float sig = 1e0;
        auto f12 = std::make_shared<SmoothnessFactor>(f12_id, _rid, variables, sig, measurement, LieType::SE2d);
        all_factors.push_back(f12);
        for (auto v : variables){
            v->add_factor(f12);
        };
    }

    while (globals.RUN){
        for (int iter=0; iter<10; iter++){
            for (int f_idx=0; f_idx<all_factors.size(); f_idx++){
                auto fac = all_factors[f_idx];
                for (auto var : fac->variables_){
                    // Read message from each connected variable
                    fac->inbox_[var->key_] = var->outbox_.at(fac->key_);
                }
                // Calculate factor potential and create outgoing messages
                fac->update_factor();
            };    
            for (int v_idx=0; v_idx<all_variables.size(); v_idx++){
                auto var = all_variables[v_idx];
                for (auto [f_key, fac] : var->factors_){
                    // Read message from each connected factor
                    var->inbox_[f_key] = fac->outbox_.at(var->key_);
                }
                // Update variable belief and create outgoing messages
                var->update_belief();
            };   
            if (iter<9) continue;
            print("Iteration:", iter);         
            print("pos 1:", iter, manif::SE2d(v1->state_).translation().eval().transpose(), manif::SE2d(v1->state_).angle() * RAD2DEG);
            print("pos 2:", iter, manif::SE2d(v2->state_).translation().eval().transpose(), manif::SE2d(v2->state_).angle() * RAD2DEG);
            // print("pos 1:", iter, manif::SO2d(v1->state_).angle() * RAD2DEG);
            // print("pos 2:", iter, manif::SO2d(v2->state_).angle() * RAD2DEG);
        }
        
        eventHandler();
        BeginDrawing();
            ClearBackground(RAYWHITE);
            BeginMode3D(graphics->camera3d);
                // Draw Ground
                for (auto var : all_variables){
                    auto pos = manif::SE2d(var->state_).translation();
                    DrawModel(graphics->landmarkModel_,
                            Vector3{(float)pos(0), (float)0.5, (float)pos(1)}, 0.1f, RED);
                    float angle = manif::SE2d(var->state_).angle();
                    float len = 0.2f;
                    DrawLine3D(Vector3{(float)pos(0), (float)0.5, (float)pos(1)},
                            Vector3{(float)pos(0) + 1.f*len*cos(angle), (float)0.5, (float)pos(1) + len*sin(angle)}, BLACK);
                }
                // Draw Robots
            EndMode3D();
        EndDrawing();  

    }


    CloseWindow();
    return 0;
}    

/*******************************************************************************/
// Load next frame
/*******************************************************************************/
void loadFrame(){
        int nf = 0;
    if (next_cam_id < cam_means.size()){

        // Create Variable for Camera next_cam_id
        Eigen::VectorXd sigma_list_cam = Eigen::VectorXd::Constant(n_dof_cam, sigma_cam);
        if (next_cam_id < 2) sigma_list_cam.setConstant(1e-3);
        auto variable_cam = std::make_shared<Variable>(next_vid_++, 0, cam_means[next_cam_id], sigma_list_cam, 0, n_dof_cam);
        cam2vid[next_cam_id] = variable_cam->v_id_;
        factorgraph->variables_[variable_cam->key_] = variable_cam;  

        // Trying to keep a sliding window of 5 keyframes 
        if (next_cam_id >= 5){
            auto cvar = factorgraph->getVar(cam2vid[cam_iter]);
            cvar->active_ = false;
            std::vector<Key> facs_to_delete{};
            for (auto [fid, fac] : cvar->factors_) facs_to_delete.push_back(fac->key_);
            for (auto f : facs_to_delete){
                cvar->delete_factor(f);
                factorgraph->factors_.erase(f);
            }

            cam_iter++; 
        }

        for (auto [lmk_id, meas] : meas_dict[next_cam_id]){
            // If this lmk_id has not had a variable created for it:
            if (!lmk2vid.count(lmk_id)){
                // Create Variable
                Eigen::VectorXd mu = lmk_means[lmk_id];// + 0.01 * Eigen::VectorXd::Random(3);
                // Eigen::VectorXd mu = Eigen::VectorXd::Zero(3);
                Eigen::VectorXd sigma_list_lmk = Eigen::VectorXd::Constant(n_dof_lmk, sigma_lmk);
                auto variable_lmk = std::make_shared<Variable>(next_vid_++, 0, mu, sigma_list_lmk, 1, n_dof_lmk);
                // Set variable as inactive and add to inactive vars.
                variable_lmk->active_ = false;
                inactive_vars[variable_lmk->v_id_] = variable_lmk;
                // Add variable to factorgraph
                factorgraph->variables_[variable_lmk->key_] = variable_lmk;    
                // Store the variable id in the landmark id lookup
                lmk2vid[lmk_id] = variable_lmk->v_id_;
                variable_lmk->update_belief();
            }

            // Create reproj factor
            auto variable_landmark = factorgraph->getVar(lmk2vid.at(lmk_id));
            std::vector<std::shared_ptr<Variable>> variables {variable_cam, variable_landmark};
            Eigen::Vector2d z = meas;
            auto factor = std::make_shared<ReprojectionFactor>(next_fid_++, 0, variables, sigma_reproj, z, K);
            nf++;
            // Set factor inactive

            factor->active_ = false;
            factor->skip_flag = true;
            
            // Add this factor to the variable's list of adjacent factors, as well as to the factorgraph
            for (auto var : factor->variables_) var->add_factor(factor);
            factorgraph->factors_[factor->key_] = factor;  
        }
        next_cam_id++;
    }

    for (auto it = inactive_vars.cbegin(), next_it = it; it != inactive_vars.cend(); it = next_it){
        ++next_it;
        auto [vid, var] = *it;
        if (var->factors_.size()>2){
            var->active_ = true;
            for (auto [fid, fac] : var->factors_){
                fac->active_ = true;
                fac->skip_flag = false;
            }
            inactive_vars.erase(it);
        } else {
            if (var->active_) print("ERROR: Active variable but has 2 or less factors");
        }
    }
    globals.SIM_MODE  = (globals.SIM_MODE==Timestep) ? SimNone : SimNone;
    print("FACTORS: ", factorgraph->factors_.size(), "new: ", nf);

}
/*******************************************************************************/
// Drawing graphics.
/*******************************************************************************/
void draw(){
    BeginDrawing();
        ClearBackground(RAYWHITE);
        BeginMode3D(graphics->camera3d);
            // Draw Ground
            drawWorldFrame();
            for (auto [vkey, var] : factorgraphs[0]->variables_){

                if (var->size_==0){
                    drawCameraVar(var);
                } else {
                    drawLandmarkVar(var);
                }
            }
            // Draw Robots
        EndMode3D();
    EndDrawing();    
};

void drawCameraVar(std::shared_ptr<Variable> cam_var){
    if (!cam_var->active_) return;
    Eigen::VectorXd axis = cam_var->mu_({3,4,5}).normalized();
    double angle = cam_var->mu_({3,4,5}).norm() * RAD2DEG;
    Eigen::VectorXd pose = -1.*so3exp(cam_var->mu_({3,4,5})).transpose() * cam_var->mu_({0,1,2});
    DrawModelWiresEx(graphics->cameraModel_, Vector3{(float)pose(0), (float)pose(1), (float)pose(2)}, 
    Vector3{(float)axis(0),(float)axis(1),(float)axis(2)}, -angle, Vector3{0.05, 0.05f, 0.05f},DARKBLUE);
}

void drawLandmarkVar(std::shared_ptr<Variable> lmk_var){
    Color col = (lmk_var->active_) ? DARKGREEN : RED;
    DrawModel(graphics->landmarkModel_, Vector3{(float)lmk_var->mu_(0), (float)lmk_var->mu_(1), (float)lmk_var->mu_(2)}, 0.01f, col);
}

void drawWorldFrame(){
    DrawModelEx(graphics->worldFrameAxisModel_, Vector3{0,0,0}, Vector3{0,0,1}, -90.f, Vector3{0.1,0.1,0.1}, RED);
    DrawModelEx(graphics->worldFrameAxisModel_, Vector3{0,0,0}, Vector3{1,0,0}, 0.f, Vector3{0.1,0.1,0.1}, GREEN);
    DrawModelEx(graphics->worldFrameAxisModel_, Vector3{0,0,0}, Vector3{1,0,0}, 90.f, Vector3{0.1,0.1,0.1}, BLUE);
}

/*******************************************************************************/
// Handles mouse and key press
/*******************************************************************************/
void eventHandler(){
    // Deal with Keyboard key press
    int key = GetKeyPressed();
    switch (key)
    {
    case KEY_ESCAPE:
            globals.RUN = false;                                                    break;
    case KEY_H:
            globals.LAST_SIM_MODE = (globals.SIM_MODE==Help) ? globals.LAST_SIM_MODE : globals.SIM_MODE;
            globals.SIM_MODE = (globals.SIM_MODE==Help) ? globals.LAST_SIM_MODE: Help;break;
    case KEY_SPACE:
            graphics->camera_transition_ = !graphics->camera_transition_;           break;
    case KEY_P:
            globals.DRAW_PATH = !globals.DRAW_PATH;                                 break;
    case KEY_R:
            globals.DRAW_INTERROBOT = !globals.DRAW_INTERROBOT;                                   break;
    case KEY_W:
            globals.DRAW_WAYPOINTS = !globals.DRAW_WAYPOINTS;                                 break;
    case KEY_ENTER:
            globals.SIM_MODE  = (globals.SIM_MODE==Timestep) ? SimNone : Timestep;  break;
    case KEY_I:
            globals.SIM_MODE  = (globals.SIM_MODE==Iterate) ? SimNone : Iterate;  print("Iterating "); break;
    case KEY_RIGHT:
    case KEY_LEFT:
            all_variables.back()->prior_factor_->z_(0) += 0.1 * (2 * (int)(key==KEY_RIGHT) - 1);
        break;
    case KEY_DOWN:
    case KEY_UP:
            all_variables.back()->prior_factor_->z_(1) += 0.1 * (2 * (int)(key==KEY_DOWN) - 1);
        break;
    case KEY_LEFT_BRACKET:
    case KEY_RIGHT_BRACKET:
    {
            auto tau = Log(all_variables.back()->prior_factor_->z_, LieType::SE2d);
            tau(2) += 0.1 * (2 * (int)(key==KEY_RIGHT_BRACKET) - 1);
            all_variables.back()->prior_factor_->z_ = Exp(tau, LieType::SE2d);
    }
        break;
            
    default:
        break;
    }

    // Mouse input handling
    Ray ray = GetMouseRay(GetMousePosition(), graphics->camera3d);
    Vector3 mouse_gnd = Vector3Add(ray.position, Vector3Scale(ray.direction, -ray.position.y/ray.direction.y));
    Vector2 mouse_pos{mouse_gnd.x, mouse_gnd.z};        // Position on the ground plane
    // Do stuff with mouse here using mouse_pos .eg:
    // if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)){
    //     print(mouse_pos.x, mouse_pos.y);
    // }

    // Update the graphics if the camera has moved
    graphics->update_camera();
}