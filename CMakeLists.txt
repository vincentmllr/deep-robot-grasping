cmake_minimum_required(VERSION 3.0.2)
project(panda_deep_grasping)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
  moveit_core
  moveit_visual_tools
  moveit_ros_planning
  moveit_ros_planning_interface
  cv_bridge
  image_transport
)
find_package(Boost REQUIRED COMPONENTS
system filesystem serialization
)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)


################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Action.msg
  Reward.msg
  State.msg
  Sample.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  ResetEnvironment.srv
  EnvironmentStep.srv
  GetBatch.srv
  GetAction.srv
  Train.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)


###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS 
    roscpp rospy std_msgs geometry_msgs message_runtime
    moveit_core
    moveit_ros_planning
    moveit_ros_planning_interface
    moveit_visual_tools
    cv_bridge
    image_transport
)


###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
 ${catkin_INCLUDE_DIRS}
)
include_directories(include  ${Boost_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS})
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS}) # at home

## Declare a C++ executable
add_executable(environment_sim src/environment_sim.cpp)
add_executable(agent_buffer src/agent_buffer.cpp)
add_executable(environment_real src/environment_real.cpp)

## Add cmake target dependencies of the executable
add_dependencies(environment_sim panda_deep_grasping_generate_messages_cpp)
add_dependencies(agent_buffer panda_deep_grasping_generate_messages_cpp)
add_dependencies(environment_real panda_deep_grasping_generate_messages_cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(environment_sim ${OpenCV_LIBRARES} ${catkin_LIBRARIES} opencv_highgui)
target_link_libraries(agent_buffer ${OpenCV_LIBRARES} ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(environment_real ${OpenCV_LIBRARES} ${catkin_LIBRARIES} opencv_highgui opencv_videoio)


#############
## Install ##
#############

## Mark executable scripts (Python etc.) for installation
catkin_install_python(PROGRAMS
  scripts/agent_model.py
  scripts/agent_main.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)