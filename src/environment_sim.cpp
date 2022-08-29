#include "ros/ros.h"
#include "geometry_msgs/Point.h"
#include "panda_deep_grasping/ResetEnvironment.h"
#include "panda_deep_grasping/EnvironmentStep.h"
#include "sensor_msgs/Image.h"
#include "gazebo_msgs/SetModelState.h"
#include "gazebo_msgs/GetModelState.h"
#include "sensor_msgs/JointState.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "control_msgs/FollowJointTrajectoryActionGoal.h"
#include "trajectory_msgs/JointTrajectoryPoint.h"
#include "control_msgs/GripperCommandActionGoal.h"
#include "franka_gripper/GraspActionGoal.h"
#include "actionlib_msgs/GoalStatusArray.h"
#include "actionlib_msgs/GoalStatus.h"
#include "actionlib_msgs/GoalID.h"
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <random>
#include <time.h>
#include <unistd.h>
#include <tf2/LinearMath/Quaternion.h>
#include <cmath>
#include <cstdlib>
#define _USE_MATH_DEFINES
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>


using namespace ros;
using namespace cv;


// Constant and counters
const int RANDOM_SEED = time(NULL);
std::default_random_engine RANDOM_GENERATOR(RANDOM_SEED);
const float MAXIMUM_CUBE_TRANSLATION = 0.1;
std::uniform_real_distribution<float> distribution_object_translation(-MAXIMUM_CUBE_TRANSLATION, MAXIMUM_CUBE_TRANSLATION);
const int LAST_IMAGES_AMOUNT = 3; // Amount of images used in state
const std::string PLANNING_GROUP_ARM = "panda_arm";
const std::string PLANNING_GROUP_HAND = "panda_hand";
const float INITIAL_POSITION_X = 0.5;
const float INITIAL_POSITION_Y = 0.0;
const float INITIAL_POSITION_Z = 0.5;
const double INITIAL_ROLL = 0;
const double INITIAL_PITCH = M_PI;
const double INITIAL_YAW = (3.0/4.0) * M_PI; 
const double GRIPPER_WIDTH_CLOSED = 0.001;
const double GRIPPER_WIDTH_OPEN = 0.08;
const double GRIPPER_FORCE = 30.0;
const double GRIPPER_EPSILON = 0.5;
const double GRIPPER_SPEED = 0.1;
const std::string ENCODING_COLOR = "rgb8"; // rgb8
const std::string ENCODING_DEPTH = ""; // 32FC1
const double JUMP_THRESHOLD = 0.0;  // Effectively disables threshold, can cause unpredictable motions, could be a safeety issue
const double EEF_STEP = 0.01;  // interpolation resolution (max step) (0.01 = 1cm)
const float REWARD_SUCCESS = 1;
const float REWARD_FAILED = 0;
const float REWARD_CONSTANT = -0.025;
const float REWARD_HEIGHT = 0.05; // factor for height([0,0.38]),should be lower than reward_success, if received with max height value (0.38) over all timesteps(40) // 0.38*40*0.005 = 0.76 < 1 !
const float REWARD_ERROR = 0;
const float OBJECT_HEIGHT = 0.05;
const float TABLE_HEIGHT = 0.02;
const float OBJECT_HEIGHT_RESET = 0.005;
float OBJECT_POSITION_RESET = 0.5 - 0.5 * OBJECT_HEIGHT;
int counter_test_save_images = 0;
const float WORKSPACE_LOWER_BOUND_X = 0.17 + 0.18;
const float WORKSPACE_UPPER_BOUND_X = 0.17 + 0.66 - 0.18;
const float WORKSPACE_LOWER_BOUND_Y = -0.15;
const float WORKSPACE_UPPER_BOUND_Y = 0.15;
const float WORKSPACE_LOWER_BOUND_Z = 0.14; // floor + distance before gripper facing downwards touches tabletop
const float WORKSPACE_UPPER_BOUND_Z = 0.55;
const float REWARD_CODE_BASIC = 0.0;
const float REWARD_CODE_SUCCESS = 1.0;
const float REWARD_CODE_FAILED = 2.0;
const float REWARD_CODE_ERROR = 3.0;
bool stop_grasp_action = false;
bool mute_grasp_check = false;
int grasp_check_counter = 0;
int grasp_check_counter_timeout = 10;
int episode_counter = 0;


// Variables to set manually
const std::string SAVE_IMAGES_FILEPATH = "../panda_deep_grasping/simulation_ws/src/panda-deep-grasping/replay_buffer_files/replay_images/";


/**
 * @brief Class for environment node that communicates with simulation. Defines publishers, subscribers and clients for interaction with gripper, arm, camera and cube and offers the reset- and the step-service. 
 */
class Environment {

  public:         

    ros::Publisher gripper_publisher;
    ros::Publisher cancel_grasp_publisher;
    ros::Subscriber gripper_status_subscriber;
    image_transport::ImageTransport it; 
    image_transport::Subscriber sub;
    ros::ServiceServer reset_server;
    ros::ServiceServer step_server;
    ros::ServiceClient cube_state_client;
    ros::ServiceClient get_model_state_client;

    moveit::planning_interface::MoveGroupInterface hand_interface; // Interface for hand
    moveit::planning_interface::MoveGroupInterface move_group_interface; // Interface for arm

    std::vector<cv::Mat> latest_images;
    std::vector<std::string> seen_goal_ids;

    std::vector<std::string> save_images();
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void open_gripper();
    void close_gripper();
    double get_rotation_radian(double rotation_degree);
    void move_to_pose(float x, float y, float z, float roll, float pitch, float yaw);
    bool move_relative(float dx, float dy, float dz, float droll, float dpitch, float dyaw);
    void move_to_inital_position();
    void graspStatusCallback(const actionlib_msgs::GoalStatusArray::ConstPtr& goal_status_array);
    float determine_reward(bool done, bool move_success, float gripper_height);
    void test_grasp();
    bool resetEnvironment(panda_deep_grasping::ResetEnvironment::Request& req, panda_deep_grasping::ResetEnvironment::Response& res);
    bool environmentStep(panda_deep_grasping::EnvironmentStep::Request& req, panda_deep_grasping::EnvironmentStep::Response& res);
    float reward_code_to_reward(float reward_code, float next_state_gripper_height);
    float get_gripper_width();
    std::vector<double> get_joint_angles(std::string move_group);

    /**
     * @brief Constructs a new Environment object and initializes publishers, subscribers and clients for 
     * interaction with gripper, arm, camera and cube and servers for the reset- and the step-service. 
     * 
     * @param node NodeHanle object so node is accessible in class
     */
    Environment(ros::NodeHandle node): move_group_interface(PLANNING_GROUP_ARM), hand_interface(PLANNING_GROUP_HAND), it(node){

      gripper_publisher = node.advertise<franka_gripper::GraspActionGoal>("/franka_gripper/grasp/goal", 1000);
      cancel_grasp_publisher = node.advertise<actionlib_msgs::GoalID>("/franka_gripper/grasp/cancel", 1000);

      sub = it.subscribe("/panda_camera/depth/image_raw", 1, &Environment::imageCallback, this);
      gripper_status_subscriber = node.subscribe("/franka_gripper/grasp/status", 1000, &Environment::graspStatusCallback, this);

      reset_server = node.advertiseService("environment_sim/reset", &Environment::resetEnvironment, this);
      ROS_INFO("environment_sim/reset advertised.");
      step_server = node.advertiseService("environment_sim/step", &Environment::environmentStep, this);
      ROS_INFO("environment_sim/step advertised.");

      cube_state_client = node.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
      get_model_state_client = node.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");

      cv::namedWindow("Camera View");
    }

};

// Class function definitions

/**
 * @brief Saves the last three images in png format. 
 * 
 * @return vector<string> filepaths of saved images
 */
std::vector<std::string> Environment::save_images() {

  ROS_DEBUG("Saving images...");

  // Get last images
  bool success = true;
  std::vector<cv::Mat> last_images = latest_images;
  if(last_images.size()>LAST_IMAGES_AMOUNT) {
    success = false;
  }

  // Preprocess images
  std::vector<cv::Mat> last_images_processed;
  for(int i=0; i<last_images.size(); i++) {

    cv::Mat image_processed = last_images.at(i);

    // Crop image
    float image_width_original = 640;
    float image_width_preffered = 480;
    float image_height = 480;
    image_processed = image_processed(Rect((image_width_original - image_width_preffered)/2, 0, image_width_preffered, image_height));

    last_images_processed.push_back(image_processed);

  }

  std::vector<std::string> image_filepaths;

  // Determine file name
  std::string replay_buffer_path = SAVE_IMAGES_FILEPATH;
  long long int now = ros::WallTime::now().toNSec();
  std::string state_identifier = std::to_string(now);

  // Save as PNG and add path
  for(int i=0; i<last_images_processed.size(); i++) {

    std::string filename = replay_buffer_path + state_identifier + "_" + std::to_string(i+1) + ".png";
    ROS_DEBUG("Filename: %s", filename.c_str());
    bool imwrite_success = cv::imwrite(filename, last_images_processed.at(i));
    ROS_DEBUG("Write image successfull? %s", imwrite_success? "True":"False");

    if(imwrite_success) {

      image_filepaths.push_back(filename);

    } else {

      success = false;

    }

  }

  ROS_DEBUG("Saving images %s.", success? "successful":"failed");

  return image_filepaths;

}

/**
 * @brief Function that is called when an image is published to the image topic. 
 * Saves the last three images in a list.
 * 
 * @param msg Pointer to image received over topic
 */
void Environment::imageCallback(const sensor_msgs::ImageConstPtr& msg) { 
  
  // Convert image for saving
  ROS_DEBUG("Depth image received with size [%d]x[%d].", msg->height, msg->width);
  ROS_DEBUG("Image encoding type of incoming image is '%s'", msg->encoding.c_str());
  cv_bridge::CvImageConstPtr cv_ptr;
  cv_ptr = cv_bridge::toCvShare(msg);
  Mat latest_image = cv::Mat(cv_ptr->image.size(), CV_8UC1);
  cv::convertScaleAbs(cv_ptr->image, latest_image, 250, 0.0);
  latest_images.push_back(latest_image);

  // Convert image for visualisation (for converted image imshow doesnt work properly)
  cv::Mat latest_image_visual = cv_bridge::toCvShare(msg, "32FC1")->image;

  // Delete oldest picture from latest_images fifo stack
  if(latest_images.size() > LAST_IMAGES_AMOUNT) latest_images.erase(latest_images.begin());
  ROS_DEBUG("Amount of images in fifo stack: %ld.", latest_images.size());

  // Show image in window
  try {

    cv::imshow("Camera View", latest_image_visual);
    cv::waitKey(30);

  } catch (cv::Exception& e) {

    ROS_ERROR("Error while showing camera image.");
    
  }

}

/**
 * @brief Retrieve gripper width using the movegroup interface of the gripper.
 * 
 * @return float width between gripper fingers
 */
float Environment::get_gripper_width() {

  std::vector<double> hand_joint_values = hand_interface.getCurrentJointValues();
  float gripper_width = hand_joint_values[0] + hand_joint_values[1];
  ROS_DEBUG("Current hand joint values: %f %f", hand_joint_values[0] , hand_joint_values[1]);

  return gripper_width;

}

/**
 * @brief Retrieve joint angles from a specific movegroup.
 * 
 * @param move_group name of the movegroup to retrieve joint angles from
 * @return vector<double> of joint angles
 */
std::vector<double> Environment::get_joint_angles(std::string move_group=PLANNING_GROUP_ARM) {

  std::vector<double>  joint_angles;

  if(move_group==PLANNING_GROUP_ARM) {

    joint_angles = move_group_interface.getCurrentJointValues();
    ROS_DEBUG("Current arm joint values size: %ld", joint_angles.size());

    return joint_angles;

  } else {

    return joint_angles;

  }

}

/**
 * @brief Function to open the gripper as a grasp-action-goal message.
 * 
 */
void Environment::open_gripper() {

  // Publish to topic to open gripper
  franka_gripper::GraspActionGoal grasp_action_goal;
  grasp_action_goal.goal.epsilon.inner = GRIPPER_EPSILON;
  grasp_action_goal.goal.epsilon.outer = GRIPPER_EPSILON;
  grasp_action_goal.goal.force = GRIPPER_FORCE;
  grasp_action_goal.goal.speed = GRIPPER_SPEED;
  grasp_action_goal.goal.width = GRIPPER_WIDTH_OPEN * 0.9;
  gripper_publisher.publish(grasp_action_goal);
  ros::Duration(GRIPPER_WIDTH_OPEN * 0.5 / GRIPPER_SPEED).sleep(); // otherwise other commands would be executed before gripper opened

  // Make sure that gripper is actually opened and return status
  std::vector<double> hand_joint_values = hand_interface.getCurrentJointValues();
  float gripper_width = hand_joint_values[0] + hand_joint_values[1];
  ROS_DEBUG("Current hand joint values: %f %f", hand_joint_values[0] , hand_joint_values[1]);
  while (gripper_width < GRIPPER_WIDTH_OPEN*0.85 && !stop_grasp_action) {

    ROS_INFO("Gripper not opened. Correct by trying to open again.");
    gripper_publisher.publish(grasp_action_goal);
    ros::Duration(1).sleep();
    hand_joint_values = hand_interface.getCurrentJointValues();
    gripper_width = hand_joint_values[0] + hand_joint_values[1];;

  }

  ROS_DEBUG("Opened gripper successful.");

}


/**
 * @brief Function to close the gripper as a grasp-action-goal message.
 * 
 */
void Environment::close_gripper() {

  // Publish to topic to open gripper
  franka_gripper::GraspActionGoal grasp_action_goal;
  grasp_action_goal.goal.epsilon.inner = GRIPPER_EPSILON;
  grasp_action_goal.goal.epsilon.outer = GRIPPER_EPSILON;
  grasp_action_goal.goal.force = GRIPPER_FORCE;
  grasp_action_goal.goal.speed = GRIPPER_SPEED;
  grasp_action_goal.goal.width = GRIPPER_WIDTH_CLOSED;
  gripper_publisher.publish(grasp_action_goal);
  ros::Duration(GRIPPER_WIDTH_OPEN * 0.5 / GRIPPER_SPEED).sleep(); // otherwise other commands would be executed before gripper opened

  // Make sure that gripper is actually closed and return status
  std::vector<double> hand_joint_values = hand_interface.getCurrentJointValues();
  float gripper_width = hand_joint_values[0] + hand_joint_values[1];
  ROS_DEBUG("Current hand joint values: %f %f", hand_joint_values[0] , hand_joint_values[1]);
  while (gripper_width >= GRIPPER_WIDTH_OPEN * 0.95 && !stop_grasp_action) {

    ROS_DEBUG("Gripper not closed. Correct by trying to close again.");
    gripper_publisher.publish(grasp_action_goal);
    hand_joint_values = hand_interface.getCurrentJointValues();
    gripper_width = hand_joint_values[0] + hand_joint_values[1];

  }
  ROS_DEBUG("Closed gripper successful.");

}

/**
 * @brief Tranforms rotation in degrees to rotation in radian.
 * 
 * @param rotation_degree rotation in degrees
 * @return double rotation in radian
 */
double Environment::get_rotation_radian(double rotation_degree) {
  return rotation_degree / 360 * 2 * M_PI;
}

/**
 * @brief Moves the endeffector to te specified position and orientation. 
 * The position is given in cartesian coordinates in the world coordinate system.
 * The orientation is given as the euler-convention roll-pitch-yaw.
 * 
 * @param x value for cartesian x-axis
 * @param y value for cartesian y-axis
 * @param z value for cartesian z-axis
 * @param roll rotation around x-axis
 * @param pitch rotation around y-axis
 * @param yaw rotation around z-axis
 */
void Environment::move_to_pose(float x, float y, float z, float roll, float pitch, float yaw) {

  // Create pose
  geometry_msgs::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  tf2::Quaternion orientation;
  orientation.setRPY(roll, pitch, yaw);
  tf2::convert(orientation, pose.orientation);

  // Compute and execute trajectory
  std::vector<geometry_msgs::Pose> waypoints = {pose};
  moveit_msgs::RobotTrajectory trajectory;
  ROS_DEBUG("Trying to compute cartesian path");
  double fraction_completed = move_group_interface.computeCartesianPath(waypoints, EEF_STEP, JUMP_THRESHOLD, trajectory);
  if(fraction_completed == 1) {

    ROS_DEBUG("%f of path computed", fraction_completed);

  }
  ROS_DEBUG("Trying to exectue cartesian path");
  move_group_interface.execute(trajectory);

}

/**
 * @brief Moves the endeffector relative according to the specified translation and rotation. 
 * The translation is given in cartesian coordinates in the world coordinate system.
 * The rotation is given as the euler-convention roll-pitch-yaw.
 * 
 * @param dx translation for cartesian x-axis
 * @param dy translation for cartesian y-axis
 * @param dz translation for cartesian z-axis
 * @param droll rotation around x-axis
 * @param dpitch rotation around y-axis
 * @param dyaw rotation around z-axis
 * 
 * @return true if movement doesnt lead out of workspace and movement was executed
 */
bool Environment::move_relative(float dx, float dy, float dz, float droll, float dpitch, float dyaw) {

  // Create pose
  geometry_msgs::Pose current_pose = move_group_interface.getCurrentPose().pose;
  geometry_msgs::Pose next_pose = current_pose;
  next_pose.position.x += dx;
  next_pose.position.y += dy;
  next_pose.position.z += dz;
  bool out_of_workspace = 
    next_pose.position.x < WORKSPACE_LOWER_BOUND_X ||
    next_pose.position.x > WORKSPACE_UPPER_BOUND_X ||
    next_pose.position.y < WORKSPACE_LOWER_BOUND_Y ||
    next_pose.position.y > WORKSPACE_UPPER_BOUND_Y ||
    next_pose.position.z < WORKSPACE_LOWER_BOUND_Z ||
    next_pose.position.z > WORKSPACE_UPPER_BOUND_Z
  ;
  if(out_of_workspace) {

    return false;

  }
  tf2::Quaternion current_orientation, rotation, new_orientation;
  tf2::convert(current_pose.orientation, current_orientation);
  rotation.setRPY(
    get_rotation_radian(droll),
    get_rotation_radian(dpitch),
    get_rotation_radian(dyaw)
  );
  new_orientation = rotation * current_orientation;
  new_orientation.normalize();
  tf2::convert(new_orientation, next_pose.orientation);

  // Compute and execute trajectory
  std::vector<geometry_msgs::Pose> waypoints = {next_pose};
  moveit_msgs::RobotTrajectory trajectory;
  double compute_success = move_group_interface.computeCartesianPath(waypoints, EEF_STEP, JUMP_THRESHOLD, trajectory);
  ROS_DEBUG("Cartesian path computed successful? %s", compute_success != -1 ? "True" : "False");
  moveit::core::MoveItErrorCode execute_success = move_group_interface.execute(trajectory);
  ROS_DEBUG("Cartesian path executed successful? %s", execute_success.val == 1 ? "True" : "False");
  bool success = (execute_success == 1);

  return success;

}

/**
 * @brief Moves endeffector to initial position.
 * 
 */
void Environment::move_to_inital_position() {

  move_to_pose(INITIAL_POSITION_X, INITIAL_POSITION_Y, INITIAL_POSITION_Z, INITIAL_ROLL, INITIAL_PITCH, INITIAL_YAW);

}

/**
 * @brief Cancels grasps that have been recalled to avoid system getting stuck at grasp command.
 * Cancels grasp if found, but skips grasps that have been found allready.
 * 
 * @param goal_status_array status of gripper actions
 */
void Environment::graspStatusCallback(const actionlib_msgs::GoalStatusArray::ConstPtr& goal_status_array) {

  actionlib_msgs::GoalID goal_id;
   
  for(actionlib_msgs::GoalStatus goal_status: (goal_status_array -> status_list)) {

    ROS_DEBUG("Status: %d", goal_status.status);

    // Check for recalled grasps
    if(!mute_grasp_check && goal_status.status == goal_status.RECALLED) {

        // Skip allready seen grasps
        auto it = std::find(seen_goal_ids.begin(), seen_goal_ids.end(), goal_status.goal_id.id);
        if (it != seen_goal_ids.end())
          continue; // Skip if error status of this id has allready been handled
        ROS_DEBUG("Text: %s, Seen ids: %ld", goal_status.text.c_str(), seen_goal_ids.size());
        ROS_INFO("Grasp action failed. Cancel action...");

        // Stop grasp action and mute check for some time
        mute_grasp_check = true;
        grasp_check_counter = grasp_check_counter_timeout;
        goal_id.id = goal_status.goal_id.id;
        seen_goal_ids.push_back(goal_status.goal_id.id);
        goal_id.stamp = goal_status.goal_id.stamp;
        stop_grasp_action = true;
        cancel_grasp_publisher.publish(goal_id);

    }

  }

  // Unmute check after some time
  if(mute_grasp_check){
      ROS_DEBUG("Grasp check muted , countdown%d", grasp_check_counter);
      grasp_check_counter--;
      if(grasp_check_counter == 0) {
        mute_grasp_check = false;
        stop_grasp_action = false;
      }
  }

}

/**
 * @brief Determines reward case of state. Cases are "successful", "failed", "error" and "basic".
 * 
 * @param done environment in terminal state?
 * @param move_success movement successful?
 * @param gripper_height height of endeffector
 * 
 * @return float for reward case of state
 */
float Environment::determine_reward(bool done, bool move_success, float gripper_height) {

  float reward = 0;

  if(done) {

      // Check for successsful grasp with cube height
      gazebo_msgs::GetModelState get_model_state;
      get_model_state.request.model_name = "cube";
      get_model_state.request.relative_entity_name = "world";
      get_model_state_client.call(get_model_state);
      float current_object_height = get_model_state.response.pose.position.z;
      ROS_DEBUG("Current object height: %f.", current_object_height);
      bool object_grasped = (current_object_height >= (TABLE_HEIGHT + OBJECT_HEIGHT));

      if(object_grasped) {

        reward = REWARD_CODE_SUCCESS;
        ROS_INFO("Grasp successful!");

      } else if(!move_success){ 

        // grasp is checked first because a movement of (0,0,0,0,0,0) with grasp=true can lead to (!move_success) but (object_grasped)
        reward = REWARD_CODE_ERROR;
        ROS_INFO("Movement error!");

      } else {
        
        reward = REWARD_CODE_FAILED;
        ROS_INFO("Grasp failed!");

      }
    
  } else {

    reward = REWARD_CODE_BASIC;

  }

  return reward;

}

/**
 * @brief Transform reward code to reward value.
 * 
 * @param reward_code code for reward case
 * @param next_state_gripper_height gripper height of next state
 * @return float reward value
 */
float Environment::reward_code_to_reward(float reward_code, float next_state_gripper_height) {

  float reward;
  switch ((int)reward_code) {
    case 0:
        reward = REWARD_HEIGHT * (INITIAL_POSITION_Z - next_state_gripper_height); // the lower the better
        break;
    case 1:
        reward = REWARD_SUCCESS;
        break;
    case 2:
        reward = REWARD_FAILED;
        break;
    case 3:
        reward = REWARD_ERROR;
        break;     
    default:
        reward = 0;
        break;
    }

    return reward;

}

/**
 * @brief Service description for reset-service. Moves robot in initial pose, places cube and returns the intial state.
 * 
 * @param req empty
 * @param res intial state
 * 
 * @return true if reset successful
 */
bool Environment::resetEnvironment(panda_deep_grasping::ResetEnvironment::Request& req,
                      panda_deep_grasping::ResetEnvironment::Response& res){

  ROS_DEBUG("Resetting environment...");

  // Move to initial pose
  move_to_inital_position();
  open_gripper();
  
  // Change cube position
  gazebo_msgs::SetModelState reset_state;
  reset_state.request.model_state.model_name = "cube";
  reset_state.request.model_state.reference_frame = "world";

  float cube_noise = episode_counter < 50 ? 0 : 0.3;
  float cube_translation_x = cube_noise * distribution_object_translation(RANDOM_GENERATOR);
  float cube_translation_y = cube_noise * distribution_object_translation(RANDOM_GENERATOR);
  reset_state.request.model_state.pose.position.x = OBJECT_POSITION_RESET + cube_translation_x;
  reset_state.request.model_state.pose.position.y = cube_translation_y;
  reset_state.request.model_state.pose.position.z = OBJECT_HEIGHT_RESET;
  bool success_cube_reset = cube_state_client.call(reset_state);
  ROS_DEBUG("Reset cube position successful? %s", success_cube_reset?"true":"false");
  episode_counter++;

  // Save images
  res.initial_state.image_filepaths = save_images();

  // Get endeffector position
  geometry_msgs::Point endeffector_position = move_group_interface.getCurrentPose().pose.position;
  res.initial_state.endeffector_position = endeffector_position;
  ROS_DEBUG("Endeffector position received: %f %f %f", endeffector_position.x, endeffector_position.y, endeffector_position.z);

  // Get gripper_width
  res.initial_state.gripper_width = get_gripper_width();
  ROS_DEBUG("Gripper_width: %f", res.initial_state.gripper_width);

  // Get joint angles
  std::vector<double> initial_state_joint_angles = get_joint_angles();
  res.initial_state.joint_angles[0] = initial_state_joint_angles.at(0);
  res.initial_state.joint_angles[1] = initial_state_joint_angles.at(1);
  res.initial_state.joint_angles[2] = initial_state_joint_angles.at(2);
  res.initial_state.joint_angles[3] = initial_state_joint_angles.at(3);
  res.initial_state.joint_angles[4] = initial_state_joint_angles.at(4);
  res.initial_state.joint_angles[5] = initial_state_joint_angles.at(5);
  res.initial_state.joint_angles[6] = initial_state_joint_angles.at(6);
  ROS_DEBUG("Joint angles: %f %f %f %f %f %f %f", 
    res.initial_state.joint_angles[0],
    res.initial_state.joint_angles[1],
    res.initial_state.joint_angles[2],
    res.initial_state.joint_angles[3],
    res.initial_state.joint_angles[4],
    res.initial_state.joint_angles[5],
    res.initial_state.joint_angles[6]
  );

  ROS_INFO("environment/reset w/ z_ee=%f",
    res.initial_state.endeffector_position.z
  );

  return true;

}

/**
 * @brief Service description for step-service. Moves robot according to received action, determines reward and next state.
 * 
 * @param req action to be executed in environemnt
 * @param res reward, next_state
 * 
 * @return true if step successful
 */
bool Environment::environmentStep(panda_deep_grasping::EnvironmentStep::Request& req,
                     panda_deep_grasping::EnvironmentStep::Response& res) {

  // Get Action from request
  float dx = req.action.relative_movement[0];
  float dy = req.action.relative_movement[1];
  float dz = req.action.relative_movement[2];
  float droll = req.action.relative_movement[3];
  float dpitch = req.action.relative_movement[4];
  float dyaw = req.action.relative_movement[5];
  int grasp = req.action.grasp;
  ROS_DEBUG("Recieved following action request: %f %f %f %f %f %f %d", dx, dy, dz, droll, dpitch, dyaw, grasp);
  
  // Move to new pose
  bool move_success = move_relative(dx, dy, dz, droll, dpitch, dyaw);
  // If movement not successful, stop execution
  if(!move_success) {
    res.done = true;
  }

  // Open or close gripper
  if(grasp) {
    close_gripper();
    res.done = true;
    move_to_inital_position();
  }

  // Get next state image filepaths
  res.next_state.image_filepaths = save_images();
  bool save_image_success = res.next_state.image_filepaths.size() > 1;

  // Get endeffector position
  geometry_msgs::Point endeffector_position = move_group_interface.getCurrentPose().pose.position;
  res.next_state.endeffector_position = endeffector_position;
  ROS_DEBUG("Endeffector position received: %f %f %f", endeffector_position.x, endeffector_position.y, endeffector_position.z);

  // Get gripper_width
  res.next_state.gripper_width = get_gripper_width();
  ROS_DEBUG("Gripper_width: %f", res.next_state.gripper_width);

  // Get joint angles
  std::vector<double> next_state_joint_angles = get_joint_angles();
  res.next_state.joint_angles[0] = next_state_joint_angles.at(0);
  res.next_state.joint_angles[1] = next_state_joint_angles.at(1);
  res.next_state.joint_angles[2] = next_state_joint_angles.at(2);
  res.next_state.joint_angles[3] = next_state_joint_angles.at(3);
  res.next_state.joint_angles[4] = next_state_joint_angles.at(4);
  res.next_state.joint_angles[5] = next_state_joint_angles.at(5);
  res.next_state.joint_angles[6] = next_state_joint_angles.at(6);
  ROS_DEBUG("Joint angles: %f %f %f %f %f %f %f", 
    res.next_state.joint_angles[0],
    res.next_state.joint_angles[1],
    res.next_state.joint_angles[2],
    res.next_state.joint_angles[3],
    res.next_state.joint_angles[4],
    res.next_state.joint_angles[5],
    res.next_state.joint_angles[6]
  );

  // Get reward for state
  res.reward.reward = determine_reward(res.done, move_success, endeffector_position.z);
  
  ROS_INFO("environment/step w/ a=%f %f %f %d, z_ee=%f, r=%f(%s), move %s, save images %s",
    dx, dy, dz, grasp,
    res.next_state.endeffector_position.z,
    reward_code_to_reward(res.reward.reward, endeffector_position.z),
    res.reward.reward == REWARD_CODE_BASIC ? 
      "basic" :
      res.reward.reward == REWARD_CODE_SUCCESS ?
        "success" :
        res.reward.reward == REWARD_CODE_FAILED ?
          "failed" : "error", 
    move_success? "successful":"failed",
    save_image_success? "successful":"failed"
  );

  return true;

}

/**
 * @brief Initializes environment_sim node, starts spinner and creates Environment object.
 * 
 * @param argc arguments
 * @param argv arguments
 * @return int status code
 */
int main(int argc, char** argv) {

  ros::init(argc, argv, "environment_sim");

  ros::NodeHandle node;

  ros::AsyncSpinner spinner(0);
  spinner.start();

  Environment environment(node);

  ros::waitForShutdown();

  cv::destroyWindow("Camera View");

  ros::shutdown();

  return 0;

}
