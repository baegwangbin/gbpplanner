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
double sigma_cam = 0.5;
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
    // print(MY_PATH);


    // Read camera params
    auto balfile_out = read_balfile(GBPPLANNER_DIR + std::string("../bundle_adjustment/data/MH01/bal.txt"));
    // auto balfile_out = read_balfile(GBPPLANNER_DIR + std::string("../bundle_adjustment/data/V103/bal.txt"));
    K = std::get<0>(balfile_out);
    cam_means = std::get<1>(balfile_out);
    lmk_means = std::get<2>(balfile_out);
    meas_dict = std::get<3>(balfile_out);
    LMK_ID_OFFSET = cam_means.size();
    
    factorgraph = std::make_shared<FactorGraph>(0);
    factorgraphs[0] = factorgraph;
    // Initially load 5 frames
    for (int i=0; i<5; i++) loadFrame();
    iterateGBP(5, INTERNAL, factorgraphs);

    while (globals.RUN){
        eventHandler();                // Capture keypresses or mouse events             
        if (globals.SIM_MODE == Timestep) loadFrame();
        iterateGBP(5, INTERNAL, factorgraphs);
        draw();
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
            // Now we are activating the variable, set it's mu to the average distance of the first cam that saw it.
            auto oldest_cam = var->factors_.begin()->second->variables_[0];
            Eigen::MatrixXd Rcw = so3exp(oldest_cam->mu_({3,4,5}));
            Eigen::Matrix4d Twc; Twc << so3exp(oldest_cam->mu_({3,4,5})), -1.*Rcw.transpose() * oldest_cam->mu_({0,1,2}),
                                        0., 0., 0., 1.;
            double init_depth = 0.; int cnt = 0;
            for (auto [fid, fac] : oldest_cam->factors_){
                if (!fac->active_) continue;
                init_depth += (fac->variables_[1]->mu_ - Twc({0,1,2},3)).norm();
                cnt++;
            }
            if (cnt>0){
                init_depth = init_depth / (float)cnt;
                Eigen::Vector3d meas_homogeneous; meas_homogeneous << var->factors_.begin()->second->z_, 1.;
                Eigen::Vector4d pc_homog; pc_homog << (K.inverse() * meas_homogeneous).normalized() * (init_depth + 0.1 * rand()/(float)RAND_MAX * init_depth), 1.;
                Eigen::VectorXd pw = Twc * pc_homog;
                Eigen::Vector3d new_mu = pw({0,1,2});
                var->change_variable_prior(new_mu);
            }
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