# How to startup system at AIFB

## Simulation environment startup

#### aifb-laptop 
1. cd ~/panda_deep_grasping/simulation_ws ; source devel/setup.bash

### aifb-nuc
1. Change filepaths:
    1. Change data file path in agent_main to "../rosworkspaces/abschlussarbeiten/panda_deep_grasping/simulation_ws/src/panda-deep-grasping/results"
    2. Change model filepath in agent_model to "../rosworkspaces/abschlussarbeiten/panda_deep_grasping/simulation_ws/src/panda-deep-grasping/models/"
    3. Change replay buffer filepath in agent_buffer to "/../rosworkspaces/abschlussarbeiten/panda_deep_grasping/simulation_ws/src/panda-deep-grasping/replay_buffer_files"
2. cd ~/rosworkspaces/abschlussarbeiten/panda_deep_grasping/simulation_ws ; source devel/setup.bash

### aifb-mlpc
1. Start VPN if not in KIT network
2. Login with remote desktop at `aifb-bis-mlpc.aifb.kit.edu`(gazebo has to be run graphically to generate pictures) 
3. cd panda_deep_grasping/simulation_ws && catkin_make && source devel/setup.bash

## Real environment startup
### aifb-nuc
1. At boot: choose Advanced Ubunto Options-> Choose Realtime kernel
2. Change filepaths:
    1. Change data file path in agent_main to "../rosworkspaces/abschlussarbeiten/panda_deep_grasping/real_ws/src/panda-deep-grasping/results"
    2. Change model filepath in agent_model to "../rosworkspaces/abschlussarbeiten/panda_deep_grasping/real_ws/src/panda-deep-grasping/models/"
    3. Change replay buffer filepath in agent_buffer to "/../rosworkspaces/abschlussarbeiten/panda_deep_grasping/real_ws/src/panda-deep-grasping/replay_buffer_files"
2. cd ~/rosworkspaces/abschlussarbeiten/panda_deep_grasping/real_ws ; source devel/setup.bash

### Startup Robot
1. Press On/Off-Button underneath desk -> robot blinks yellow
2. Unlock black button next to robot
3. When robot lights constantly yellow: Open Franka Emika Desk(FED) in Borwser https://172.16.0.2/desk/
4. Unlock Joints in Sidebar in FED
5. Press Activate FCI in Menu in FED
6. Launch: roslaunch panda_moveit_config panda_control_moveit_rviz.launch launch_rviz:=true robot_ip:=172.16.0.2 load_gripper:=true

### Shutdown Robot
1. Shutdown in FED
2. Wait for control fans to shut off (loading symbol in FED)
3. Switch off with onoff button underneath desk