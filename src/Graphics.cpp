/**************************************************************************************/
// Copyright (c) 2023 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed under MIT license (see LICENSE for details)
/**************************************************************************************/
#include <Graphics.h>

/**************************************************************************/
// Graphics class that deals with the nitty-gritty of display.
// Camera is also included here. You can set different camera positions/trajectories
// and then during simulation cycle through them using the SPACEBAR

// Please note: Raylib camera defines the world with positive X = right, positive Z = down, and positive Y = out-of-plane
// But in our work we use the standard convention of positive X = right, positive Y = down, and positive Z = into-plane
/**************************************************************************/
Graphics::Graphics(Image obstacleImg) : obstacleImg_(ImageCopy(obstacleImg)){
    if (!globals.DISPLAY) return;

    // Camera is defined by a forward vector (target - position), as well as an up vector (see raylib for more info)
    // These are vectors for each camera transition. Cycle through them in the simulation with the SPACEBAR
    camera_positions_ = {Vector3{-0.5,1., -0.5}};
    camera_ups_ = {Vector3{0.,1.,0.}};
    camera_targets_ = {Vector3{1.,0.,1.}};

    camera3d.position = camera_positions_[camera_idx_];
    camera3d.target = camera_targets_[camera_idx_];
    camera3d.up = camera_ups_[camera_idx_];             // Camera up vector
    camera3d.fovy = 60.0f;                              // Camera field-of-view Y
    camera3d.projection = CAMERA_PERSPECTIVE;           // Camera mode type

    // Load basic lighting shader
    lightShader_ = LoadShader((globals.ASSETS_DIR+"shaders/base_lighting.vs").c_str(),
                            (globals.ASSETS_DIR+"shaders/lighting.fs").c_str());

    // Get some required shader locations
    lightShader_.locs[SHADER_LOC_VECTOR_VIEW] = GetShaderLocation(lightShader_, "viewPos");
    
    // Ambient light level (some basic lighting)
    int ambientLoc = GetShaderLocation(lightShader_, "ambient");
    float temp[4] = {0.1f, 0.1f, 0.1f, 1.0f};
    SetShaderValue(lightShader_, ambientLoc, temp, SHADER_UNIFORM_VEC4);

    // Assign our lighting shader to robot model
    cameraModel_ = LoadModelFromMesh(GenMeshCone(1., -1., 4.));
    cameraModel_.transform = MatrixMultiply(MatrixTranslate(0., -2., 0.),MatrixRotateXYZ(Vector3{90.f*DEG2RAD, 45.f*DEG2RAD, 0.f})) ;
    cameraModel_.materials[0].shader = lightShader_;
    cameraModel_.materials[0].maps[0].color = WHITE;

    // Height map
    Mesh mesh = GenMeshHeightmap(obstacleImg_, (Vector3){ 1.f*globals.WORLD_SZ, 1.f*globals.ROBOT_RADIUS, 1.f*globals.WORLD_SZ }); // Generate heightmap mesh (RAM and VRAM)
    ImageColorInvert(&obstacleImg_);                     // TEXTURE REQUIRES OBSTACLES ARE BLACK
    texture_img_ = LoadTextureFromImage(obstacleImg_); 

    landmarkModel_ = LoadModelFromMesh(GenMeshSphere(1., 10.0f, 10.0f));               // Load model from generated mesh
    landmarkModel_.materials[0].shader = lightShader_;
    landmarkModel_.materials[0].maps[0].color = WHITE;

    worldFrameAxisModel_ = LoadModelFromMesh(GenMeshCylinder(0.05, 0.5, 6.));
    worldFrameAxisModel_.materials[0].maps[0].color = WHITE;    
    // Create lights
    Light lights[MAX_LIGHTS] = { 0 };
    Vector3 target = camera3d.target;
    Vector3 position = Vector3{target.x-10,target.y-20,target.z-10};
    lights[0] = CreateLight(LIGHT_POINT, position, target, LIGHTGRAY, lightShader_);                            
}

Graphics::~Graphics(){
    UnloadTexture(texture_img_);
};

/******************************************************************************************/
// Use captured mouse input and keypresses and modify the camera view.
// Also transition between camera viewframes if necessary.
/******************************************************************************************/
void Graphics::update_camera()
{
    float zoomscale = IsKeyDown(KEY_LEFT_SHIFT) ? 1. :0.5;
    float zoom = -(float)GetMouseWheelMove() * zoomscale;
    CameraMoveToTarget(&camera3d, zoom);
    if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE))
    {
        Vector2 del = GetMouseDelta();
        // FOR UP {0,0,-1} and TOWARDS STRAIGHT DOWN
        if (IsKeyDown(KEY_LEFT_SHIFT)){
            CameraPitch(&camera3d, -del.y*0.05, true, true, true);                
            // Rotate up direction around forward axis
            camera3d.up = Vector3RotateByAxisAngle(camera3d.up, Vector3{0.,1.,0.}, -0.05*del.x);                
            Vector3 forward = Vector3Subtract(camera3d.target, camera3d.position);
            forward = Vector3RotateByAxisAngle(forward, Vector3{0.,1.,0.}, -0.05*del.x);
            camera3d.position = Vector3Subtract(camera3d.target, forward);
        } else if (IsKeyDown(KEY_LEFT_CONTROL)){
            float zoom = del.y*0.1;
            CameraMoveToTarget(&camera3d, zoom);
        } else {
            // Camera movement
            CameraMoveRight(&camera3d, -del.x*0.1, true);
            Vector3 D = GetCameraUp(&camera3d); D.y = 0.;
            D = Vector3Scale(Vector3Normalize(D), del.y*0.1);
            camera3d.position = Vector3Add(camera3d.position, D);
            camera3d.target = Vector3Add(camera3d.target, D);
        }
    }
    if (camera_transition_){
        int camera_transition_time = 100;
        if (camera_clock_==camera_transition_time){
            camera_transition_ = false;
            camera_idx_ = (camera_idx_+1)%camera_positions_.size();
            camera_clock_ = 0;
        }
        camera3d.position = Vector3Lerp(camera_positions_[camera_idx_], camera_positions_[(camera_idx_+1)%camera_positions_.size()], (camera_clock_%camera_transition_time)/(float)camera_transition_time);
        camera3d.up = Vector3Lerp(camera_ups_[camera_idx_], camera_ups_[(camera_idx_+1)%camera_ups_.size()], (camera_clock_%camera_transition_time)/(float)camera_transition_time);
        camera3d.target = Vector3Lerp(camera_targets_[camera_idx_], camera_targets_[(camera_idx_+1)%camera_targets_.size()], (camera_clock_%camera_transition_time)/(float)camera_transition_time);
        camera_clock_++;

    }    
}