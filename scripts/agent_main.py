#!/usr/bin/env python
import rospy
import csv
import sys
from random import randint, random, uniform
import numpy as np
import matplotlib.pyplot as plt
from panda_deep_grasping.msg import Sample, Action, State
from panda_deep_grasping.srv import ResetEnvironment, ResetEnvironmentRequest, EnvironmentStep, EnvironmentStepRequest, Train, TrainRequest, GetAction, GetActionRequest, GetBatch, GetBatchRequest


LOOP_FREQUENCY = 10 # in Hz
NUMBER_OF_EPISODES = 100000
NUMBER_OF_TIMESTEPS = 25 # 40
OBJECT_HEIGHT = 0.05
TABLE_HEIGHT_SIM = 0.02
TABLE_HEIGHT_REAL = -0.039 # tabletop is under robot base
DISTANCE_FLANGE_TCP = 0.1
BATCH_SIZE = 16
MAXIMUM_CARTESIAN_MOVEMENT = 0.02
MAXIMUM_ROTATIONAL_MOVEMENT = 2
MAXIMUM_CARTESIAN_MOVEMENT_RANDOM = 0.5 * MAXIMUM_CARTESIAN_MOVEMENT
MAXIMUM_ROTATIONAL_MOVEMENT_RANDOM = 0.5 * MAXIMUM_ROTATIONAL_MOVEMENT
MAXIMUM_CARTESIAN_MOVEMENT_NOISE = 0.002
MAXIMUM_ROTATIONAL_MOVEMENT_NOISE = 0.2
MINIMUM_GRIPPER_HEIGHT_SIM = OBJECT_HEIGHT + TABLE_HEIGHT_SIM + DISTANCE_FLANGE_TCP -0.005 + 0.02 # Technically its the minimum flange height
MINIMUM_GRIPPER_HEIGHT_REAL = OBJECT_HEIGHT + TABLE_HEIGHT_REAL + DISTANCE_FLANGE_TCP + 0.01 # Technically its the minimum flange height
REWARD_CODE_BASIC = 0.0
REWARD_CODE_SUCCESS = 1.0
REWARD_CODE_FAILED = 2.0
REWARD_CODE_ERROR = 3.0
REWARD_SUCCESS = 1
REWARD_FAILED = 0
REWARD_CONSTANT = -0.025
REWARD_HEIGHT = 0.05 # factor for height([0,0.38]),should be lower than reward_success, if received with max height value (0.38) over all timesteps(40) // 0.38*40*0.005 = 0.76 < 1 !
REWARD_ERROR = 0
INITIAL_POSITION_Z = 0.5


# Variables to set maunally
START_EPISODE = 0
save_data_version = "2.0"
RESULTS_FILE_PATH = "../panda_deep_grasping/simulation_ws/src/panda-deep-grasping/results"


class Agent_Model_Interface():
    '''
    Interface to communicate with agent_model node.
    '''

    def __init__(self, minimum_gripper_height):
        '''
        Initialize interface to communicate with agent_model node by creating service clients for train an get_action ROS-services.
        '''

        self.minimum_gripper_height = minimum_gripper_height
        
        # Service Clients
        rospy.wait_for_service('agent_model/train')
        self.agent_actor_critic_train_service = rospy.ServiceProxy('agent_model/train', Train)
        self.agent_actor_critic_get_action_service = rospy.ServiceProxy('agent_model/get_action', GetAction)


    def train(self, batch):
        '''
        Calls train service.

        Arguments:
        `batch`: batch of sample ROS-messages
        '''
        try:
            response = self.agent_actor_critic_train_service(batch)
        except rospy.ServiceException as e:
            rospy.loginfo('Service call failed: %s'%e)


    def get_action(self, state, episode, action_type="actor"):
        '''
        Determine action as simple action, actor action or actor action with noise. Calls get_action service if need be.

        Arguments:
        `state`: state message
        `episode`: number of current episode
        `action_type`: type of action to perform. Either simple, actor or actor_noise
        '''

        actor = False
        noise = False
        simple = False

        if(action_type == "actor"):
            actor = True
        elif(action_type == "simple"):
            simple = True
        elif(action_type == "actor_noise"):
            actor = True
            noise = True
        else:
            actor = True

        action = Action()

        if(simple) :

            simple_noise = 0.6 

            dx = simple_noise * np.random.uniform(-MAXIMUM_CARTESIAN_MOVEMENT_RANDOM, MAXIMUM_CARTESIAN_MOVEMENT_RANDOM)
            dy = simple_noise * np.random.uniform(-MAXIMUM_CARTESIAN_MOVEMENT_RANDOM, MAXIMUM_CARTESIAN_MOVEMENT_RANDOM)
            dz = -MAXIMUM_CARTESIAN_MOVEMENT # != 0, to avoid failed movement
            droll = 0
            dpitch = 0
            dyaw = simple_noise * np.random.uniform(-MAXIMUM_ROTATIONAL_MOVEMENT_RANDOM, MAXIMUM_ROTATIONAL_MOVEMENT_RANDOM)

            action.relative_movement[0]= dx
            action.relative_movement[1]= dy
            action.relative_movement[2]= dz
            action.relative_movement[3]= droll
            action.relative_movement[4]= dpitch
            action.relative_movement[5]= dyaw

            grasp = (state.endeffector_position.z < self.minimum_gripper_height)
            action.grasp = grasp

        elif(actor):

            try:
                request = GetActionRequest(state)
                response = self.agent_actor_critic_get_action_service(request)
                action = response.action
            except rospy.ServiceException as e:
                rospy.loginfo('Service call failed: %s'%e)

            rospy.logdebug(f'Determined actor action: {action.relative_movement[0]}, {action.relative_movement[1]}, {action.relative_movement[2]}, {action.relative_movement[3]}, {action.relative_movement[4]}, {action.relative_movement[5]}, {action.grasp}')

            if(noise):

                dx_noise = np.random.uniform(-MAXIMUM_CARTESIAN_MOVEMENT_NOISE, MAXIMUM_CARTESIAN_MOVEMENT_NOISE)
                dy_noise = np.random.uniform(-MAXIMUM_CARTESIAN_MOVEMENT_NOISE, MAXIMUM_CARTESIAN_MOVEMENT_NOISE)
                dz_noise = np.random.uniform(-MAXIMUM_CARTESIAN_MOVEMENT_NOISE, MAXIMUM_CARTESIAN_MOVEMENT_NOISE)
                droll_noise = np.random.uniform(-MAXIMUM_ROTATIONAL_MOVEMENT_NOISE, MAXIMUM_ROTATIONAL_MOVEMENT_NOISE)
                dpitch_noise = np.random.uniform(-MAXIMUM_ROTATIONAL_MOVEMENT_NOISE, MAXIMUM_ROTATIONAL_MOVEMENT_NOISE)
                dyaw_noise = np.random.uniform(-MAXIMUM_ROTATIONAL_MOVEMENT_NOISE, MAXIMUM_ROTATIONAL_MOVEMENT_NOISE)

                dx = action.relative_movement[0] + dx_noise
                dy = action.relative_movement[1] + dy_noise
                dz = action.relative_movement[2] + dz_noise
                droll = action.relative_movement[3] + droll_noise
                dpitch = action.relative_movement[4] + dpitch_noise
                dyaw = action.relative_movement[5] + dyaw_noise

                relative_movement = [dx, dy, dz, droll, dpitch, dyaw]
                
                for i in range(len(relative_movement)):

                    coordinate = relative_movement[i]
               
                    if(i < 3):
                        maximum_movement = MAXIMUM_CARTESIAN_MOVEMENT
                    else:
                        maximum_movement = MAXIMUM_ROTATIONAL_MOVEMENT

                    if(coordinate > maximum_movement):
                        coordinate = maximum_movement
                    elif (coordinate < -maximum_movement):
                        coordinate = -maximum_movement

                    relative_movement[i] = coordinate
                
                if(action.grasp == 1) and (np.random.uniform() <= 0.5):
                    action.grasp = 0

                action.relative_movement = [
                    relative_movement[0],
                    relative_movement[1],
                    relative_movement[2],
                    relative_movement[3],
                    relative_movement[4],
                    relative_movement[5]
                ]

                rospy.logdebug(f'get_action: {action.relative_movement[0]}, {action.relative_movement[1]}, {action.relative_movement[2]}, {action.relative_movement[3]}, {action.relative_movement[4]}, {action.relative_movement[5]}, {action.grasp}')
            
            # Deactivate action variables to simpify problem: only cartesian movement and rotation around z-axis used
            # Activate action variables to make the problem more complex
            action.relative_movement = [
                action.relative_movement[0],
                action.relative_movement[1],
                action.relative_movement[2],
                0,
                0,
                action.relative_movement[5]
            ]
            grasp = (state.endeffector_position.z < self.minimum_gripper_height)
            action.grasp = grasp

        rospy.logdebug(f'get_action: {action.relative_movement[0]}, {action.relative_movement[1]}, {action.relative_movement[2]}, {action.relative_movement[3]}, {action.relative_movement[4]}, {action.relative_movement[5]}, {action.grasp}')

        return action


class Agent_Buffer_Interface():
    '''
    Interface to communicate with agent_buffer node.
    '''

    def __init__(self):
        '''
        Initialize interface to communicate with agent_buffer node by creating publisher for add_sample ROS-topic and service clients for get_batch ROS-service.
        '''
        
        # Publisher
        self.agent_buffer_add_sample = rospy.Publisher('agent_buffer/add_sample', Sample, queue_size=1000)

        # Serivce Clients
        rospy.wait_for_service('agent_buffer/get_batch')
        self.agent_buffer_get_batch_service = rospy.ServiceProxy('agent_buffer/get_batch', GetBatch)


    def add_sample(self, state, action, reward, next_state):
        '''
        Publish sample to add_sample topic.

        Arguments:
        `state`: state of sample
        `action`: action of sample
        `reward`: reward of sample
        `next_state`: next_state of sample
        '''

        sample = Sample(state=state, action=action, reward=reward, next_state=next_state)

        self.agent_buffer_add_sample.publish(sample)


    def get_batch(self):
        '''
        Calls get_batch service.

        Returns:
        `batch`: batch of sample ROS-messages
        '''
        
        try:
            response = self.agent_buffer_get_batch_service()
            return response.batch
        except rospy.ServiceException as e:
            rospy.loginfo('Service call failed: %s'%e)


class Environment_Sim_Interface():
    '''
    Interface to communicate with environment_sim node.
    '''

    def __init__(self):
        '''
        Initialize interface to communicate with environment_sim node by creating service clients for reset an step ROS-services.
        '''
        
        # Check if gazebo is running
        rospy.wait_for_service('/gazebo/reset_simulation')

        # Clients
        rospy.wait_for_service('environment_sim/reset')
        self.reset_environment_service = rospy.ServiceProxy('environment_sim/reset', ResetEnvironment)
        rospy.wait_for_service('environment_sim/step')
        self.environment_step_service = rospy.ServiceProxy('environment_sim/step', EnvironmentStep)


    def reset(self):
        '''
        Calls the reset service to reset the environment.

        Returns:
        `initial_state`: initial state of environment
        '''

        try:
            response = self.reset_environment_service()
            return response.initial_state
        except rospy.ServiceException as e:
            rospy.loginfo('Service call failed: %s'%e)


    def step(self, action):
        '''
        Calls the step service to execute an action in the environment.

        Arguments:
        `action`: action to perform during environment step

        Returns:
        `next_state`: state of environment after step
        `reward`: code for reward received for action
        `done`: whether the environment is in a terminal state        
        '''

        try:
            request = EnvironmentStepRequest(action=action)
            response = self.environment_step_service(request)
            next_state = response.next_state
            reward = response.reward
            done = response.done
            return next_state, reward, done
        except rospy.ServiceException as e:
            rospy.loginfo('Service call failed: %s'%e)


class Environment_Real_Interface():
    '''
    Interface to communicate with environment_sim node.
    '''

    def __init__(self):
        '''
        Initialize interface to communicate with environment_real node by creating service clients for reset an step ROS-services.
        '''
        
        # Clients
        rospy.wait_for_service('environment_real/reset')
        self.reset_environment_service = rospy.ServiceProxy('environment_real/reset', ResetEnvironment)
        rospy.wait_for_service('environment_real/step')
        self.environment_step_service = rospy.ServiceProxy('environment_real/step', EnvironmentStep)


    def reset(self):
        '''
        Calls the reset service to reset the environment.

        Returns:
        `initial_state`: initial state of environment
        '''

        try:
            response = self.reset_environment_service()
            return response.initial_state
        except rospy.ServiceException as e:
            rospy.loginfo('Service call failed: %s'%e)


    def step(self, action):
        '''
        Calls the step service to execute an action in the environment.

        Arguments:
        `action`: action to perform during environment step

        Returns:
        `next_state`: state of environment after step
        `reward`: code for reward received for action
        `done`: whether the environment is in a terminal state        
        '''

        try:
            request = EnvironmentStepRequest(action=action)
            response = self.environment_step_service(request)
            next_state = response.next_state
            reward = response.reward
            done = response.done
            return next_state, reward, done
        except rospy.ServiceException as e:
            rospy.loginfo('Service call failed: %s'%e)


# Help functions

def reward_from_code(reward_code, next_state_gripper_height):
    '''
    Calculates the reward based on the reward code received from the environment step.

    Arguments:
    `reward_code`: code received from the environment step
    `next_state_gripper_height`: gripper height to calculate reward based on height

    Returns:
    `reward`: value of reward received for action
    '''

    reward = 0

    if reward_code==0 :

        reward = REWARD_HEIGHT * (INITIAL_POSITION_Z - next_state_gripper_height) # the lower the better

    elif reward_code==1 :

        reward = REWARD_SUCCESS

    elif reward_code==2 :

        reward = REWARD_FAILED

    elif reward_code==3 :

        reward = REWARD_ERROR

    else :

        reward = 0

    return reward


def get_average(list, maximum_amount=0):
    '''
    Calculates the average of a last x entries of a list. If `maximum amount` is zero, calcultes the average over all entries.

    Arguments:
    `list`: list of values
    `maximum_amount`: amount of entries to use for average

    Returns:
    `list_average`: average of the last x entries of the list
    '''
    
    if(maximum_amount == 0):

        list_average = np.mean(list)

    else:

        list_average = np.mean(list[-maximum_amount:])

    return list_average


def add_leading_zeros(number, amount_zeros):
    '''
    Adds leading zeros to a number.

    Arguments:
    `number`: numeric value
    `amount_zeros`: number of zeros to add

    Returns:
    `number_with_leading_zeros`: number with leading zeros added
    '''

    number_str = str(number)
    number_with_leading_zeros = number_str.zfill(amount_zeros)

    return number_with_leading_zeros


def get_training_action_type(probability_actor, episode):
    '''
    Determines action type during training: Tries actor every 50 episodes, otherwise either simple actions or actor actions with noise based.

    Arguments:
    `probability_actor`: probability with which to choose the actor with noise action, if not actor is choosen
    `episode`: current episode

    Returns:
    `action_type`: type of action to perform during episode: "actor", "actor_noise" or "simple"
    '''

    if episode%50 == 0:

        return "actor"
    
    else:

        if np.random.uniform() < probability_actor:
                return "actor_noise"
        else:
                return "simple"


def main():
    '''
    Initialize agent_main node for handling of training process. Before the training is started creates interfaces for agent_buffer, agent_model and environment nodes.
    '''
    
    rospy.init_node('agent_main')

    # Determine mode
    training = True
    training_light_data_collection = False
    training_light_learning = False
    evaluation_sim = False
    evaluation_real = False

    if (len(sys.argv) > 1):
        rospy.loginfo(f'len(argv)={len(sys.argv)}, argv[0]={sys.argv[0]}, argv[1]={sys.argv[1]}')
        for arg in sys.argv :
            rospy.loginfo(f"arg reveived: {arg}")
        arg = sys.argv[1]
        training = (arg == "training") | (arg == "training_light_data_collection") | (arg == "training_light_learning")
        training_light_data_collection = (arg == "training_light_data_collection")
        training_light_learning = (arg == "training_light_learning")
        evaluation_sim = (arg == "evaluation_sim")
        evaluation_real = (arg == "evaluation_real")

    # Create interfaces to other nodes
    if evaluation_real:
        agent = Agent_Model_Interface(MINIMUM_GRIPPER_HEIGHT_REAL)
    else: 
        agent = Agent_Model_Interface(MINIMUM_GRIPPER_HEIGHT_SIM)

    if not training_light_learning and not evaluation_real:
        environment = Environment_Sim_Interface()
    if evaluation_real:
        environment = Environment_Real_Interface()
    
    if training:
            replay_buffer = Agent_Buffer_Interface()

    # Initialize variables
    rate = rospy.Rate(10) # 10hz
    updates_pro_episode = 10
    episode_returns = []
    episode_returns_actor_actions = []
    episode_returns_noise_actions = []
    episode_returns_simple_actions = []
    episode_grasp_success_list = []
    average_episode_returns = []
    episode_durations = []
    average_episode_return = 0
    training_action_mode = "random"; # One of simple, actor_noise, random(simple or actor_noise)
    total_timesteps = 0
    total_batch_updates = 0
    simple_action_probability = 1.0
    actor_action_probability = 0.0

    # Open file to write data
    if training:
        data_file = open(RESULTS_FILE_PATH + "/data_training" + save_data_version + ".txt", "w")
    elif evaluation_sim:
        data_file = open(RESULTS_FILE_PATH + "/data_evaluation_sim" + save_data_version + ".txt", "w")
    else:
        data_file = open(RESULTS_FILE_PATH + "/data_evaluation_real" + save_data_version + ".txt", "w")
    data_writer = csv.writer(data_file)
    data_writer.writerow(["episode", "episode_action_type", "episode_return", "average_episode_return_noise_actions", "average_episode_return_actor_actions", "episode_grasp_success", "#timesteps", "#episodes", "duration_episode", "average_episode_duration"])
    

    # Start training process
    if not rospy.is_shutdown():
        
        start_time_training = rospy.Time.now()
        rospy.loginfo(f'Starting training at {start_time_training.to_sec()}s')

        for episode in range(NUMBER_OF_EPISODES):

            if episode < START_EPISODE:
                continue
            
            # Reset episode varibles
            start_time_episode = rospy.Time.now()
            episode_return = 0
            episode_grasp_success = False
            done = False
            timestep = 0
            simple_action_probability = 1 if episode < 50 else 0.1
            actor_action_probability = 1 - simple_action_probability
            episode_action_type = get_training_action_type(actor_action_probability, episode) if training else "actor"

            # Training step without data collection
            if training_light_learning :
                batch = replay_buffer.get_batch()
                agent.train(batch)
                total_batch_updates += 1
                episode_duration = rospy.Time.now() - start_time_episode
                rospy.loginfo(f'E{add_leading_zeros(episode+1, 3)}/{add_leading_zeros(NUMBER_OF_EPISODES, 3)} finished in {int(episode_duration.to_sec())}s; #updates: {total_batch_updates}')
                continue

            # Reset environment
            state = environment.reset()

            while not done :
                
                start_time_timestep = rospy.Time.now()
                
                # Determine Action
                action = agent.get_action(state, episode, episode_action_type)

                # Act and retrieve changes in simulated environment
                next_state, reward, done = environment.step(action)
                                
                # Send training sample to replay buffer // TODO if not evaluation
                if training: replay_buffer.add_sample(state, action, reward, next_state)

                timestep_duration = rospy.Time.now() - start_time_timestep
   
                episode_return += reward_from_code(reward.reward, next_state.endeffector_position.z)
                state = next_state

                timestep += 1
                if (timestep >= NUMBER_OF_TIMESTEPS):
                    done = True

                if reward.reward == REWARD_CODE_SUCCESS:
                    episode_grasp_success = True

            # Training steps after each episode if training
            if(not training_light_data_collection and not evaluation_sim and not episode < 50):
                for update in range(updates_pro_episode):
                    # Get batch from replay buffer
                    batch = replay_buffer.get_batch()
                    # Train actor and critic net
                    agent.train(batch)
                    total_batch_updates += 1

            # Collect and write data
            episode_duration = rospy.Time.now() - start_time_episode
            episode_durations.append(episode_duration)
            average_episode_duration = get_average(episode_durations)
            total_timesteps += timestep
            episode_returns.append(episode_return)
            if episode_action_type=="actor_noise" : episode_returns_noise_actions.append(episode_return)
            if episode_action_type=="actor" : episode_returns_actor_actions.append(episode_return)
            average_episode_return_noise_actions = get_average(episode_returns_noise_actions, 100)
            average_episode_return_actor_actions = get_average(episode_returns_actor_actions, 100)
            episode_actor_grasp_success = 1 if ((episode_action_type=="actor_noise" or episode_action_type=="actor") and episode_grasp_success) else 0
            average_episode_returns.append(average_episode_return)
            episode_grasp_success_list.append(episode_actor_grasp_success)
            relative_grasp_success = get_average(episode_grasp_success_list, 100)
            rospy.loginfo(f'E{add_leading_zeros(episode+1, 3)}/{add_leading_zeros(NUMBER_OF_EPISODES, 3)} finished in {int(episode_duration.to_sec())}s w/ R={episode_return}, R_avg={average_episode_return_noise_actions}, total batch updates:{total_batch_updates}, total timesteps:{total_timesteps}, action type:{episode_action_type}, rel. grasp success: {relative_grasp_success}')
            data_writer.writerow([episode+1, episode_action_type, episode_return, average_episode_return_noise_actions, average_episode_return_actor_actions, episode_grasp_success, timestep, NUMBER_OF_EPISODES, episode_duration.to_sec(), average_episode_duration.to_sec()])
    
    data_file.close()
    

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
