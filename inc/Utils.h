/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed under MIT license (see LICENSE for details)
/**************************************************************************************/
#pragma once
#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <tuple>
#include <Eigen/Core>
#include <Eigen/Dense>

/*******************************************************************************/
// Easy print statement, just use like in python: print(a, b, "blah blah");
// If printing Eigen::Vector or Matrix but after accessing incline functions, you will need to call .eval() first:
// Eigen::VectorXd myVector{{0,1,2}}; print(myVector({0,1}).eval())
/*******************************************************************************/
template <typename T> void print(const T& t) {
    std::cout << t << std::endl;
}
template <typename First, typename... Rest> void print(const First& first, const Rest&... rest) {
    std::cout << first << ", ";
    print(rest...); // recursive call using pack expansion syntax
}

/*******************************************************************************/
// This function draws the FPS and time on the screen, as well as the help screen
/*******************************************************************************/
void draw_info( uint32_t time_cnt);

/*******************************************************************************/
// This function allows you to time events in real-time
// Usage: 
// auto start = std::chrono::steady_clock::now();
// std::cout << "Elapsed(us): " << since(start).count() << std::endl;
/*******************************************************************************/
template <
    class result_t   = std::chrono::microseconds,
    class clock_t    = std::chrono::high_resolution_clock,
    class duration_t = std::chrono::microseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
};

inline std::map<std::string, float> read_camera_params(std::string filename){
    auto cam_params = std::map<std::string, float>{};
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream split(line);
            std::vector<std::string> splitted;
            for (std::string each; std::getline(split, each, ' '); splitted.push_back(each));
            if (splitted.size() > 0) {
                if (splitted[0].compare("Camera1.fx:") == 0){
                    cam_params["f1"] = std::stof(splitted[1]);
                } else if (splitted[0].compare("Camera1.fy:") == 0){
                    cam_params["f2"] = std::stof(splitted[1]);
                } else if (splitted[0].compare("Camera1.cx:") == 0){
                    cam_params["c1"] = std::stof(splitted[1]);
                } else if (splitted[0].compare("Camera1.cy:") == 0){
                    cam_params["c2"] = std::stof(splitted[1]);
                } else if (splitted[0].compare("Camera1.k1:") == 0){
                    cam_params["k1"] = std::stof(splitted[1]);
                } else if (splitted[0].compare("Camera1.k2:") == 0){
                    cam_params["k2"] = std::stof(splitted[1]);
                } else if (splitted[0].compare("Camera1.p1:") == 0){
                    cam_params["p1"] = std::stof(splitted[1]);
                } else if (splitted[0].compare("Camera1.p2:") == 0){
                    cam_params["p2"] = std::stof(splitted[1]);
                } else if (splitted[0].compare("Camera.width:") == 0){
                    cam_params["width"] = std::stof(splitted[1]);
                } else if (splitted[0].compare("Camera.height:") == 0){
                    cam_params["height"] = std::stof(splitted[1]);
                }
            }
        }
        file.close();
    }
    return cam_params;
};
inline std::vector<long> read_KFfile(std::string filename){
    auto timestamps = std::vector<long>{};
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream split(line);
            std::vector<std::string> splitted;
            for (std::string each; std::getline(split, each, ' '); splitted.push_back(each));
            if (splitted.size() > 1) {
                timestamps.push_back(std::stol(splitted[1].substr(0, 12)));
            }
        }
        file.close();
    }
    return timestamps;
};

using balfile_result_type = std::tuple< Eigen::MatrixXd, std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, 
                                        std::map<int, std::map<int, Eigen::VectorXd>>, 
                                        int, int, int, std::vector<Eigen::VectorXd>
                                      >;
inline balfile_result_type read_balfile(std::string filename){
    balfile_result_type result;

    std::ifstream file(filename);
    std::vector<std::string> splitted;
    int n_keyframes;
    int n_points;
    int n_edges;
    std::vector<Eigen::VectorXd> cam_means{};
    std::vector<Eigen::VectorXd> lmk_means{};
    std::vector<Eigen::VectorXd> measDistort{};
    std::map<int, std::map<int, Eigen::VectorXd>> meas_dict{};
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(3,3);

    if (file.is_open()) {
        std::string line;
        while (true) {
            std::getline(file, line);
            std::istringstream split(line);
            splitted.clear(); for (std::string each; std::getline(split, each, ' '); splitted.push_back(each));
            if (splitted.size() && (splitted[0].compare("#") != 0)) {
                n_keyframes = std::stoi(splitted[0]);
                n_points = std::stoi(splitted[1]);
                n_edges = std::stoi(splitted[2]);
                break;
            }
        }

        std::getline(file, line);
        std::istringstream split(line);
        splitted.clear(); for (std::string each; std::getline(split, each, ' '); splitted.push_back(each));   
        K(0,0) = std::stof(splitted[0]); K(1,1) = std::stof(splitted[1]);
        K(0,2) = std::stof(splitted[2]); K(1,2) = std::stof(splitted[3]);
        K(2,2) = 1.;

        for (int i=0; i<n_edges; i++){
            std::getline(file, line);
            std::istringstream split(line);
            splitted.clear(); for (std::string each; std::getline(split, each, ' '); splitted.push_back(each));   
            meas_dict[std::stoi(splitted[0])][std::stoi(splitted[1])] = Eigen::VectorXd{{(double)std::stof(splitted[2])}, {(double)std::stof(splitted[3])}};

            if (splitted.size() > 4) measDistort.push_back(Eigen::VectorXd{{(double)std::stof(splitted[4])}, {(double)std::stof(splitted[5])}});
        }

        for (int i=0; i<n_keyframes; i++){
            Eigen::VectorXd vec = Eigen::VectorXd::Zero(6);
            for (int j=0; j<6; j++){
                std::getline(file, line);
                std::istringstream split(line);
                splitted.clear(); for (std::string each; std::getline(split, each, ' '); splitted.push_back(each));  
                vec(j) = std::stof(splitted[0]);
            }
            cam_means.push_back(vec);
        }
        for (int i=0; i<n_points; i++){
            Eigen::VectorXd vec = Eigen::VectorXd::Zero(3);
            for (int j=0; j<3; j++){
                std::getline(file, line);
                std::istringstream split(line);
                splitted.clear(); for (std::string each; std::getline(split, each, ' '); splitted.push_back(each));  
                vec(j) = std::stof(splitted[0]);
            }
            lmk_means.push_back(vec);
        }

        file.close();
    }

    return std::make_tuple(K, cam_means, lmk_means, meas_dict, n_keyframes, n_points, n_edges, measDistort);
};