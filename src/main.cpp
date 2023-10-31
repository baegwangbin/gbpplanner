/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed under MIT license (see LICENSE for details)

// Define all parameters in the appropriate config file (default: config/config.json)
/**************************************************************************************/
#define RLIGHTS_IMPLEMENTATION // needed to be defined once for the lights shader
#include <iostream>
#include <Utils.h>

#include <DArgs.h>

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
double sigma_lmk = 100;
double sigma_cam = 1;
double sigma_reproj = 100.;
std::vector<Eigen::VectorXd> cam_means;
std::vector<Eigen::VectorXd> lmk_means;
std::map<int, std::map<int, Eigen::VectorXd>> meas_dict;
std::map<int, std::shared_ptr<Variable>> inactive_vars{};
std::map<int, int> num_measurements_lmk{}; // lmk_id, num
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



    // Read camera params
    // auto [n_keyframes, n_points, n_edges, cam_means_temp, lmk_means_temp, measurements, meas_distort, measurements_camIDs, \
    //         measurements_lIDs, K_temp, meas_dict_temp] = read_balfile("/home/ap721/code/visIMU/bundle_adjustment/data/MH01/bal.txt");
    auto balfile_out = read_balfile("/home/ap721/code/visIMU/bundle_adjustment/data/MH01/bal.txt");
    cam_means = std::get<3>(balfile_out);
    lmk_means = std::get<4>(balfile_out);
    meas_dict = std::get<10>(balfile_out);
    K = std::get<9>(balfile_out);
    LMK_ID_OFFSET = cam_means.size();
    
    factorgraph = std::make_shared<FactorGraph>(0);
    factorgraphs[0] = factorgraph;
        // Eigen::VectorXd sigma_list_lmk = Eigen::VectorXd::Constant(n_dof_lmk, sigma_lmk);
        // for (auto v : lmk_means){
        //     auto variable_lmk = std::make_shared<Variable>(next_vid_++, 0, v, sigma_list_lmk, 1, n_dof_lmk);
        //     factorgraph->variables_[variable_lmk->key_] = variable_lmk;    

        // }
    
    while (globals.RUN){
        eventHandler();                // Capture keypresses or mouse events             
        if (globals.SIM_MODE == Timestep) loadFrame();
        if (globals.SIM_MODE == Iterate) iterateGBP(10, INTERNAL, factorgraphs);
        draw();
    }

    CloseWindow();
    return 0;
}    

/*******************************************************************************/
// Load next frame
/*******************************************************************************/
void loadFrame(){
    if (next_cam_id < cam_means.size()){
        print("Loading Frame ", next_cam_id, cam_means[next_cam_id].transpose());
        Eigen::VectorXd sigma_list_cam = Eigen::VectorXd::Constant(n_dof_cam, sigma_cam);
        auto variable_cam = std::make_shared<Variable>(next_vid_++, 0, cam_means[next_cam_id], sigma_list_cam, 0, n_dof_cam);
        factorgraph->variables_[variable_cam->key_] = variable_cam;    

        next_cam_id++;

        Eigen::VectorXd sigma_list_lmk = Eigen::VectorXd::Constant(n_dof_lmk, sigma_lmk);
        for (auto [lmk_id, meas] : meas_dict[next_cam_id]){
            // If new mid not found in created v ids
            if (!factorgraph->variables_.count(Key{0, lmk_id + LMK_ID_OFFSET})){
                Eigen::VectorXd mu = lmk_means[lmk_id] + 0.1*Eigen::VectorXd::Random(3); // TODO
                // Eigen::VectorXd mu = Eigen::VectorXd::Zero(3);
                auto variable_lmk = std::make_shared<Variable>(lmk_id + LMK_ID_OFFSET, 0, mu, sigma_list_lmk, 1, n_dof_lmk);
                variable_lmk->active_ = false;
                inactive_vars[variable_lmk->v_id_] = variable_lmk;
                factorgraph->variables_[variable_lmk->key_] = variable_lmk;    
            }
            std::vector<std::shared_ptr<Variable>> variables {variable_cam, factorgraph->getVar(lmk_id + LMK_ID_OFFSET)};
            auto factor = std::make_shared<ReprojectionFactor>(next_fid_++, 0, variables, sigma_reproj, meas, K);
            print("VAR BETWEEN ", variable_cam->v_id_, variables.back()->v_id_, meas.transpose());
            factor->active_ = false;
            
            // Add this factor to the variable's list of adjacent factors, as well as to the robot's list of factors
            for (auto var : factor->variables_){
                var->add_factor(factor);
            }
            factorgraph->factors_[factor->key_] = factor;  
        }
    }
    for (auto it = inactive_vars.begin(); it != inactive_vars.end();)
    {
        auto [vid, var] = *it;
        if (var->factors_.size()>2){
            var->active_ = true;
            for (auto [fid, fac] : var->factors_){
                fac->active_ = true;
            }
            it = inactive_vars.erase(it);
        } else {
            ++it;
        }
    }
    print("!", inactive_vars.size(), factorgraph->variables_.size());
    globals.SIM_MODE  = (globals.SIM_MODE==Timestep) ? SimNone : SimNone;

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
                // if (vkey.node_id_>5) continue;

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
    Eigen::VectorXd axis = cam_var->mu_({3,4,5}).normalized();
    double angle = cam_var->mu_({3,4,5}).norm() * RAD2DEG;
    Eigen::VectorXd pose = -1.*so3exp(cam_var->mu_({3,4,5})).transpose() * cam_var->mu_({0,1,2});
    DrawModelEx(graphics->landmarkModel_, Vector3{(float)pose(0), (float)pose(1), (float)pose(2)}, 
    Vector3{(float)axis(0),(float)axis(1),(float)axis(2)}, angle, Vector3{0.01, 0.01f, 0.01f},DARKGREEN);
}

void drawLandmarkVar(std::shared_ptr<Variable> lmk_var){
    DrawModel(graphics->landmarkModel_, Vector3{(float)lmk_var->mu_(0), (float)lmk_var->mu_(1), (float)lmk_var->mu_(2)}, 0.01f, RED);
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