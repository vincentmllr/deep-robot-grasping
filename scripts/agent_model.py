#!/usr/bin/env python
import math
import rospy
from std_msgs.msg import String
from panda_deep_grasping.msg import Action, State
from panda_deep_grasping.srv import GetAction, GetActionRequest, GetActionResponse, Train, TrainResponse, GetBatch, GetBatchResponse, TrainRequest
from random import randint
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, MaxPool2D, Add
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2


# Constants
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
SOFT_UPDATE_FACTOR = 0.001 # aka tau
REWARD_DISCOUNT_FACTOR = 0.99 # for future rewards, aka gamma
INPUT_IMAGE_HEIGHT = 480
INPUT_IMAGE_WIDTH = 480
NUMBER_OF_STATE_VALUES_WITHOUT_IMAGES = 7+3+1 # 7 joint angles, 3 ee coordinates, 1 gripper width
NUMBER_OF_ACTION_VALUES = 6+1 # Six relative movement values and grasp action value
ACTION_MAXIMUM = 2.0 # Maximum cartesian(cm) or rotational movement (deg)
MINIMUM_GRIPPER_HEIGHT = 0.1 # Technically its the minimum flange height
WORKSPACE_LOWER_BOUND_X = 0.17 + 0.18; # distance gripper robot + additional workspace limitation to 0.3m
WORKSPACE_UPPER_BOUND_X = 0.17 + 0.66 - 0.18; # distance gripper robot + table length - additional workspace limitation to 0.3m
WORKSPACE_LOWER_BOUND_Y = -0.15; # additional workspace limitation to 0.3m
WORKSPACE_UPPER_BOUND_Y = 0.15; # additional workspace limitation to 0.3m
WORKSPACE_LOWER_BOUND_Z = 0.14; # floor + distance before gripper facing downwards touches tabletop
WORKSPACE_UPPER_BOUND_Z = 0.55; # 0.55m
GRIPPER_WIDTH_CLOSED = 0.001
GRIPPER_WIDTH_OPEN = 0.08


# Variables to set manually
ONLY_ACTOR = False
MODEL_FILEPATH = "../panda_deep_grasping/simulation_ws/src/panda-deep-grasping/models/"
LOAD_PRETRAINED_MODEL = True
PRETRAINED_MODEL_VERSION = "1.2.7"
VERSION_NAME = "2.0"


class Actor() :
    '''
    The actor network that chooses an action depending on the state and receives batches to train the network.
    '''

    def __init__(
        self,
        learning_rate,
        action_maximum,
        model_filepath,
        load_pretrained_model,
        pretrained_model_version,
        number_of_state_values_without_images,
        input_image_height,
        input_image_width,
        number_of_action_values,
        version_name,
        workspace_lower_bound_x,
        workspace_upper_bound_x,
        workspace_lower_bound_y,
        workspace_upper_bound_y,
        workspace_lower_bound_z,
        workspace_upper_bound_z,
        gripper_width_open

    ):
        '''
        Initializes the actor network with optimizer, create service server for get_action and service client for get_batch.

        Arguments:
        `learning_rate`: rate at which the weights are adjusted by
        `action_maximum`: maximum action value
        `model_filepath`: path where to save an load the model
        `load_pretrained_model`: whether to load an allready existing model
        `pretrained_model_version`: version of the model to load, if version is "3.0" file "actor0.3.h5" will be loaded
        `number_of_state_values_without_images`: number of state variables that are not the pixels of an image
        `input_image_height`: height of input images
        `input_image_width`: widht of input images
        `number_of_action_values`: number of action variables
        `version_name`: version under which the model is saved
        `workspace_lower_bound_x`: lower bound of workspace in x-direction
        `workspace_upper_bound_x`: upper bound of workspace in x-direction
        `workspace_lower_bound_y`: lower bound of workspace in y-direction
        `workspace_upper_bound_y`: upper bound of workspace in y-direction
        `workspace_lower_bound_z`: lower bound of workspace in z-direction
        `workspace_upper_bound_z`: upper bound of workspace in z-direction
        `gripper_width_open`: width of the gripper fingers when gripper is opened
        '''

        # Offered services
        self.get_action_service = rospy.Service("agent_model/get_action", GetAction, self.get_action)

        # Used services
        self.get_batch = rospy.ServiceProxy("agent_buffer/get_batch", GetBatch)

        # Initialize variables
        self.learning_rate = learning_rate
        self.action_maximum = action_maximum
        self.saving_counter = 1
        self.saving_frequency = 200 # Every x learning step the model is saved
        self.model_filepath = model_filepath
        self.model_name = "actor"
        self.load_pretrained_model = load_pretrained_model
        self.autosave_version = 0
        self.pretrained_model_version = pretrained_model_version
        self.number_of_state_values_without_images = number_of_state_values_without_images
        self.input_image_height = input_image_height
        self.input_image_width = input_image_width
        self.number_of_action_values = number_of_action_values
        self.version_name = version_name
        self.workspace_lower_bound_x = workspace_lower_bound_x
        self.workspace_upper_bound_x = workspace_upper_bound_x
        self.workspace_lower_bound_y = workspace_lower_bound_y
        self.workspace_upper_bound_y = workspace_upper_bound_y
        self.workspace_lower_bound_z = workspace_lower_bound_z
        self.workspace_upper_bound_z = workspace_upper_bound_z
        self.gripper_width_open = gripper_width_open

        #Initialize actor network
        self.model = self.initialize_model()
        rospy.loginfo(f'Initialized actor: {self.model.summary()}')
        if self.load_pretrained_model:
            model_path = self.model_filepath + self.model_name + pretrained_model_version + ".h5"
            self.model.load_weights(model_path)
            rospy.loginfo(f'Loaded actor from file: {self.model.summary()}')

        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)


    def initialize_model(self):
        '''
        Initializes the actor neural network model.

        Returns:
        `model`: neural network model of actor
        '''

        # Input network without images
        input_layer_state_rest = layers.Input(shape=(self.number_of_state_values_without_images))
        dense_layer_state_rest = layers.Dense(64, activation="relu")(input_layer_state_rest)
        reshape_layer_state_rest = layers.Reshape((1, 1, 64))(dense_layer_state_rest)
        # Input network images
        input_layer_image_1 = layers.Input(shape=(self.input_image_height, self.input_image_width, 1))
        input_layer_image_2 = layers.Input(shape=(self.input_image_height, self.input_image_width, 1))
        input_layer_image_3 = layers.Input(shape=(self.input_image_height, self.input_image_width, 1))
        concat_input_images_layer = layers.Concatenate()([
            input_layer_image_1,
            input_layer_image_2,
            input_layer_image_3
        ])
        convolution_layer_images_1 = Conv2D(64, 6, 2, activation='relu')(concat_input_images_layer)
        max_pool_layer_images_1 = MaxPool2D(3)(convolution_layer_images_1)
        convolution_layer_images_2 = Conv2D(64, 5, 1, activation='relu')(max_pool_layer_images_1)
        convolution_layer_images_3 = Conv2D(64, 5, 1, activation='relu')(convolution_layer_images_2)
        convolution_layer_images_4 = Conv2D(64, 5, 1, activation='relu')(convolution_layer_images_3)
        convolution_layer_images_5 = Conv2D(64, 5, 1, activation='relu')(convolution_layer_images_4)
        convolution_layer_images_6 = Conv2D(64, 5, 1, activation='relu')(convolution_layer_images_5)
        convolution_layer_images_7 = Conv2D(64, 5, 1, activation='relu')(convolution_layer_images_6)
        max_pool_layer_images_3 = MaxPool2D(3)(convolution_layer_images_7)
        convolution_layer_1 = Conv2D(64, 3, 1, activation='relu')(max_pool_layer_images_3)
        convolution_layer_2 = Conv2D(64, 3, 1, activation='relu')(convolution_layer_1)
        convolution_layer_3 = Conv2D(64, 3, 1, activation='relu')(convolution_layer_2)
        convolution_layer_4 = Conv2D(64, 3, 1, activation='relu')(convolution_layer_3)
        max_pool_layer_1 = layers.MaxPool2D(2)(convolution_layer_4)
        convolution_layer_7 = Conv2D(64, 3, 1, activation='relu')(max_pool_layer_1)
        convolution_layer_8 = Conv2D(64, 3, 1, activation='relu')(convolution_layer_7)
        # Concat inputs
        concat_input_layer = layers.Concatenate(axis=1)(
            [
                reshape_layer_state_rest,
                convolution_layer_8
            ]
        )
        dense_layer_1 = layers.Dense(64, activation="relu")(concat_input_layer)
        flatten_layer_1 = Flatten()(dense_layer_1)
        dense_layer_2 = layers.Dense(64, activation="relu")(flatten_layer_1)
        output_layer = layers.Dense(
            self.number_of_action_values,
            activation="tanh",
            kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        )(dense_layer_2)

        model = tf.keras.Model(
            [
                input_layer_state_rest,
                input_layer_image_1,
                input_layer_image_2,
                input_layer_image_3
            ],
                output_layer
        )

        return model


    def state_to_tensor(self, state):
        '''
        Convert a state ROS-msg to a tensor

        Arguments:
        `state`: state ROS-msg-object

        Returns:
        `state_tensor`: tensor of state variables
        '''

        state_rest_tensor = []

        # Normalize joint angles from -pi to pi to [0,1]
        state_rest_tensor.append(tf.convert_to_tensor((state.joint_angles[0] + math.pi) / (2 * math.pi)))
        state_rest_tensor.append(tf.convert_to_tensor((state.joint_angles[1] + math.pi) / (2 * math.pi)))
        state_rest_tensor.append(tf.convert_to_tensor((state.joint_angles[2] + math.pi) / (2 * math.pi)))
        state_rest_tensor.append(tf.convert_to_tensor((state.joint_angles[3] + math.pi) / (2 * math.pi)))
        state_rest_tensor.append(tf.convert_to_tensor((state.joint_angles[4] + math.pi) / (2 * math.pi)))
        state_rest_tensor.append(tf.convert_to_tensor((state.joint_angles[5] + math.pi) / (2 * math.pi)))
        state_rest_tensor.append(tf.convert_to_tensor((state.joint_angles[6] + math.pi) / (2 * math.pi)))
        # Normalize endeffector position from -bounds to bounds to [0,1]
        state_rest_tensor.append(tf.convert_to_tensor((state.endeffector_position.x - self.workspace_lower_bound_x)) / (self.workspace_upper_bound_x-self.workspace_lower_bound_x))
        state_rest_tensor.append(tf.convert_to_tensor((state.endeffector_position.y - self.workspace_lower_bound_y)) / (self.workspace_upper_bound_y-self.workspace_lower_bound_y))
        state_rest_tensor.append(tf.convert_to_tensor((state.endeffector_position.z - self.workspace_lower_bound_z)) / (self.workspace_upper_bound_z-self.workspace_lower_bound_z))
        # Normalize gripper_width to [0,1]
        state_rest_tensor.append(tf.convert_to_tensor((state.gripper_width / self.gripper_width_open)))

        state_tensor = [state_rest_tensor]

        for image_filepath in state.image_filepaths:

            image_raw = cv2.imread(image_filepath)
            image_gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
            image_normalized = image_gray / 255
            image_processed = tf.expand_dims(image_normalized, -1)

            state_tensor.append(image_processed)

        return state_tensor


    def action_to_tensor(self, action):
        '''
        Convert an action ROS-msg to a tensor

        Arguments:
        `action`: action ROS-msg-object

        Returns:
        `action_tensor`: tensor of action variables
        '''

        action_tensor = []

        # Transform relative cartesian movements from -0.02 to 0.02m to -1 to 1
        dx = action.relative_movement[0] * 100.0 / 2.0
        dy = action.relative_movement[1] * 100.0 / 2.0
        dz = action.relative_movement[2] * 100.0 / 2.0
        # Transform relative rotational movements from -2 to 2deg to -1 to 1
        droll = action.relative_movement[3] / 2.0
        dpitch = action.relative_movement[4] / 2.0
        dyaw = action.relative_movement[5] / 2.0
        # Transform grasp from boolean to -1 to 1
        if(action.grasp):
            grasp = 1.0
        else:
            grasp = -1.0

        # Convert action variables to tensors
        action_tensor.append(tf.convert_to_tensor(dx))
        action_tensor.append(tf.convert_to_tensor(dy))
        action_tensor.append(tf.convert_to_tensor(dz))
        action_tensor.append(tf.convert_to_tensor(droll))
        action_tensor.append(tf.convert_to_tensor(dpitch))
        action_tensor.append(tf.convert_to_tensor(dyaw))
        action_tensor.append(tf.convert_to_tensor(grasp))

        rospy.logdebug(action_tensor)

        return action_tensor


    def get_state_tensor(self, state):
        '''
        Get state tensor of a batch with only one state. Used to create input for model when retrieving action of only one state.

        Arguments:
        `state`: state ROS-msg-object

        Returns:
        `state_tensor`: tensor of state variables as input for model
        '''

        state_rest_array = np.zeros((1, NUMBER_OF_STATE_VALUES_WITHOUT_IMAGES))
        state_image_1_array = np.zeros((1, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1))
        state_image_2_array = np.zeros((1, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1))
        state_image_3_array = np.zeros((1, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1))

        state_tensor = self.state_to_tensor(state)

        state_rest_array[0] = state_tensor[0]
        state_image_1_array[0] = state_tensor[1]
        state_image_2_array[0] = state_tensor[2]
        state_image_3_array[0] = state_tensor[3]

        state_rest_tensor = tf.convert_to_tensor(state_rest_array)
        state_image_1_tensor = tf.convert_to_tensor(state_image_1_array)
        state_image_2_tensor = tf.convert_to_tensor(state_image_2_array)
        state_image_3_tensor = tf.convert_to_tensor(state_image_3_array)

        state_tensor = [state_rest_tensor, state_image_1_tensor, state_image_2_tensor, state_image_3_tensor]

        return state_tensor


    def get_batch_tensors(self, batch):
        '''
        Convert batch of samples to seperate batches of tensors for state, action, reward, next_state and not_done

        Arguments:
        `batch`: batch of sample ROS-messages

        Returns:
        `state_batch_tensor`: batch of state tensors
        `action_batch_tensor`: batch of action tensors
        `reward_batch_tensor`: batch of reward tensors
        `next_state_batch_tensor`: batch of next_state tensors
        `not_done_batch_tensor`: batch of not_done tensors. true, if next_state is not a terminal state
        '''

        batch_size = len(batch)
        rospy.logdebug(f'batch size: {batch_size}')

        state_rest_array = np.zeros((batch_size, self.number_of_state_values_without_images))
        state_image_1_array = np.zeros((batch_size, self.input_image_height, self.input_image_width, 1))
        state_image_2_array = np.zeros((batch_size, self.input_image_height, self.input_image_width, 1))
        state_image_3_array = np.zeros((batch_size, self.input_image_height, self.input_image_width, 1))
        action_batch_array = np.zeros((batch_size, self.number_of_action_values))
        reward_batch_array = np.zeros((batch_size, 1))
        next_state_rest_array = np.zeros((batch_size, NUMBER_OF_STATE_VALUES_WITHOUT_IMAGES))
        next_state_image_1_array = np.zeros((batch_size, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1))
        next_state_image_2_array = np.zeros((batch_size, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1))
        next_state_image_3_array = np.zeros((batch_size, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1))
        not_done_batch_array = np.zeros((batch_size, 1))

        for s in range(batch_size):

            state = batch[s].state
            action = batch[s].action
            reward = batch[s].reward
            next_state = batch[s].next_state
            if(
                reward.reward == 1 or
                reward.reward == 0 or
                action.grasp or
                next_state.endeffector_position.x < 0.35 or
                next_state.endeffector_position.x > 0.65 or
                next_state.endeffector_position.y < -0.15 or
                next_state.endeffector_position.y > 0.15 or
                next_state.endeffector_position.z < 0.14 or
                next_state.endeffector_position.z > 0.55

            ):
                not_done = 1.0 # 0.0 to enable using the done variable at training
            else:
                not_done = 1.0

            state_tensor = self.state_to_tensor(state)
            action_tensor = self.action_to_tensor(action)
            reward_tensor = tf.convert_to_tensor(reward.reward)
            next_state_tensor = self.state_to_tensor(next_state)
            not_done_tensor = tf.convert_to_tensor(not_done)

            state_rest_array[s] = state_tensor[0]
            state_image_1_array[s] = state_tensor[1]
            state_image_2_array[s] = state_tensor[2]
            state_image_3_array[s] = state_tensor[3]
            action_batch_array[s] = action_tensor
            reward_batch_array[s] = reward_tensor
            next_state_rest_array[s] = next_state_tensor[0]
            next_state_image_1_array[s] = next_state_tensor[1]
            next_state_image_2_array[s] = next_state_tensor[2]
            next_state_image_3_array[s] = next_state_tensor[3]
            not_done_batch_array[s] = not_done_tensor

        state_rest_tensor = tf.convert_to_tensor(state_rest_array)
        state_image_1_tensor = tf.convert_to_tensor(state_image_1_array)
        state_image_2_tensor = tf.convert_to_tensor(state_image_2_array)
        state_image_3_tensor = tf.convert_to_tensor(state_image_3_array)
        state_batch_tensor = [state_rest_tensor, state_image_1_tensor, state_image_2_tensor, state_image_3_tensor]
        action_batch_tensor = tf.convert_to_tensor(action_batch_array)
        reward_batch_tensor = tf.convert_to_tensor(reward_batch_array)
        reward_batch_tensor = tf.cast(reward_batch_tensor, dtype=tf.float32)
        next_state_rest_tensor = tf.convert_to_tensor(next_state_rest_array)
        next_state_image_1_tensor = tf.convert_to_tensor(next_state_image_1_array)
        next_state_image_2_tensor = tf.convert_to_tensor(next_state_image_2_array)
        next_state_image_3_tensor = tf.convert_to_tensor(next_state_image_3_array)
        next_state_batch_tensor = [next_state_rest_tensor, next_state_image_1_tensor, next_state_image_2_tensor, next_state_image_3_tensor]
        not_done_batch_tensor = tf.convert_to_tensor(not_done_batch_array)
        not_done_batch_tensor = tf.cast(not_done_batch_tensor, dtype=tf.float32)

        return state_batch_tensor, action_batch_tensor, reward_batch_tensor, next_state_batch_tensor, not_done_batch_tensor


    def train_actor_critic(self, request):
        '''
        Definition of train-ROS-Service. Receives a batch of samples, updates and saves actor and critic models.

        Requests:
        `batch`: batch of sample ROS-messages
        '''

        rospy.logdebug("Started training step.")
        start_time_train_step = rospy.Time.now()

        batch_msgs = request.batch

        # Stop if no batch received
        if(len(batch_msgs) == 0) :
            return 0

        # Saving models
        start_time_save_model = rospy.Time.now()

        if(self.saving_counter % self.saving_frequency == 0):

            rospy.loginfo("Saving actor and critic models.")
            self.save_model(self.model, self.model_name + VERSION_NAME)
            self.save_model(self.target.model, self.target.model_name + VERSION_NAME)
            self.save_model(self.critic.model, self.critic.model_name + VERSION_NAME)
            self.save_model(self.critic.target.model, self.critic.target.model_name + VERSION_NAME)

        if(self.saving_counter % 5000 == 0):

            rospy.loginfo("Auto-Saving actor and critic models.")
            self.save_model(self.model, self.model_name + VERSION_NAME + '.' + str(self.autosave_version))
            self.save_model(self.target.model, self.target.model_name + VERSION_NAME + '.' + str(self.autosave_version))
            self.save_model(self.critic.model, self.critic.model_name + VERSION_NAME + '.' + str(self.autosave_version))
            self.save_model(self.critic.target.model, self.critic.target.model_name + VERSION_NAME + '.' + str(self.autosave_version))

            self.autosave_version += 1

        self.saving_counter += 1
        save_model_duration = rospy.Time.now() - start_time_save_model

        # Get batch tensors from batch msgs
        state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = self.get_batch_tensors(batch_msgs)
        rospy.logdebug(f'Batch received: {state_batch}, {action_batch}, {reward_batch}, {next_state_batch}')

        # Actor and Critic training step
        start_time_train_actor = rospy.Time.now()
        actor_loss = 0.0
        actor_loss = self.train(state_batch)
        train_actor_duration = rospy.Time.now() - start_time_train_actor
        start_time_train_critic = rospy.Time.now()
        critic_error = self.critic.train(batch_msgs)
        train_critic_duration = rospy.Time.now() - start_time_train_critic

        # Actor and critic target training step
        start_time_train_actor_target = rospy.Time.now()
        self.target.update_weights()
        train_actor_target_duration = rospy.Time.now() - start_time_train_actor_target
        start_time_train_critic_target = rospy.Time.now()
        self.critic.target.update_weights()
        train_critic_target_duration = rospy.Time.now() - start_time_train_critic_target

        train_step_duration = rospy.Time.now() - start_time_train_step

        rospy.loginfo(f"agent_actor_critc/train: updated actor({train_actor_duration.to_sec()}s, loss={actor_loss}), critic({train_critic_duration.to_sec()}s, error={critic_error}), actor_target({train_actor_target_duration.to_sec()}s), critic_target({train_critic_target_duration.to_sec()}s) in {train_step_duration.to_sec()}s.")

        return TrainResponse()


    def train(self, state_batch_tensor):
        '''
        Update actor weights by calculating and applying the gradient of the actor loss

        Argmuents:
        `state_batch_tensor`: batch of state tensors

        Returns:
        `actor_loss`: actor_loss of batch
        '''

        state_batch = state_batch_tensor

        # Calculate gradient of actor model
        with tf.GradientTape() as tape:

            # a(s) = a
            actor_actions = self.get_action_batch(state_batch)
            rospy.logdebug(f'Actor actions: {actor_actions.shape}: {actor_actions} {actor_actions.dtype}')
            # Q(s,a) / Expected future reward
            critic_values = self.critic.get_value_batch(state_batch, actor_actions)
            rospy.logdebug(f'Critic values {critic_values.shape}: {critic_values} {critic_values.dtype}')
            # Loss function for actor, the bigger the critic_values, the smaller the loss
            actor_loss = -tf.math.reduce_mean(critic_values)
            rospy.logdebug(f'Actor loss {actor_loss.shape}: {actor_loss} {actor_loss.dtype}')

        # Derivative of the loss function with respect to the actor weights
        actor_loss_gradient = tape.gradient(actor_loss, self.model.trainable_variables)
        rospy.logdebug(f'Actor loss gradient: {actor_loss_gradient}')

        # Change weights in direction of gradient of loss function
        rospy.logdebug(f'Actor weights before actor update: {self.model.trainable_variables}')
        rospy.logdebug(f'Actor target weights before actor update: {self.target.model.trainable_variables}')
        self.optimizer.apply_gradients(
            zip(actor_loss_gradient, self.model.trainable_variables)
        )
        rospy.logdebug(f'Actor weights after actor update: {self.model.trainable_variables}')
        rospy.logdebug(f'Actor target weights after actor update: {self.target.model.trainable_variables}')

        test = False
        if test:
            state_batch.pop() # gradient tape seems to add an element at the end of state_batch
            actor_actions_test = self.get_action_batch(state_batch)
            rospy.logdebug(f'Actor actions: {actor_actions_test.shape}: {actor_actions_test} {actor_actions_test.dtype}')
            critic_values_test = self.critic.get_value_batch(state_batch, actor_actions_test)
            actor_loss_test = -tf.math.reduce_mean(critic_values_test)
            rospy.logdebug(f'Actor loss {actor_loss.shape}: {actor_loss} {actor_loss.dtype}')
            rospy.logdebug(f'Actor loss test(<actor_loss?) {actor_loss_test.shape}: {actor_loss_test} {actor_loss_test.dtype}')
            rospy.loginfo(f'Actor update successful(lowered actor loss)? {actor_loss_test < actor_loss}, difference={actor_loss_test - actor_loss}')

        rospy.logdebug(f"Finished actor update with gradient: {actor_loss_gradient}")

        return actor_loss


    def get_action(self, request):
        '''
        Definition of get_action-ROS-Service. Receives a state and calculates the action using the actor.

        Requests:
        `state`: state ROS-message

        Responds:
        `action`: action ROS-message calculated by the actor
        '''

        state = request.state

        # Create action depending on state
        action = self.model(self.get_state_tensor(state), training=True)
        rospy.logdebug(f"Action before tranformation: {action}")
        action = tf.gather(action, 0)

        # Transform cartesian movements from -1 to 1 to -0.02 to 0.02m
        dx = tf.gather(action, 0) / 100.0 * self.action_maximum
        dy = tf.gather(action, 1) / 100.0 * self.action_maximum
        dz = tf.gather(action, 2) / 100.0 * self.action_maximum
        # Transform rotational movements from -1 to 1 to -2 to 2deg
        droll = tf.gather(action, 3) * self.action_maximum
        dpitch = tf.gather(action, 4) * self.action_maximum
        dyaw = tf.gather(action, 5) * self.action_maximum
        # Transform grasp from -1 to 1 to boolean
        grasp = (tf.gather(action, 6) > 0)

        rospy.loginfo(f"actor/get_action: x:{dx}, y:{dy}, z:{dz}, droll:{droll}, dpitch:{dpitch}, dyaw:{dyaw}, grasp:{grasp}.")

        response = GetActionResponse()
        response.action.relative_movement[0] = dx
        response.action.relative_movement[1] = dy
        response.action.relative_movement[2] = dz
        response.action.relative_movement[3] = droll
        response.action.relative_movement[4] = dpitch
        response.action.relative_movement[5] = dyaw
        response.action.grasp = grasp

        return response


    def get_action_batch(self, state_batch):
        '''
        Calculates the actions of a batch of states using the actor.

        Arguments:
        `state_batch`: batch of states

        Returns:
        `action_batch`: batch of actions calculated by the actor
        '''

        action_batch = self.model(state_batch, training=True)

        return action_batch


    def set_target(self, actor_target):
        '''
        Sets the actor target to make it accessible during training step.

        Arguments:
        `actor_target`: actor target object
        '''
        self.target = actor_target


    def set_critic(self, critic):
        '''
        Sets the critic to make it accessible during training step and creates service server for train-ROS-service

        Arguments:
        `critic`: critic object
        '''
        self.critic = critic
        self.train_service = rospy.Service("agent_model/train", Train, self.train_actor_critic)


    def save_model(self, model, name):
        '''
        Saves a model under a given name at model filepath

        Arguments:
        `model`: model object
        `name`: model name
        '''

        filenname = self.model_filepath + name
        rospy.logdebug(f'Filename of saved {name} model: {filenname + ".h5"}.')

        # Save the weights
        model.save_weights(filenname + ".h5")


class Actor_Target() :
    '''
    The actor target network that is a copy of the actor network, which is updated with a soft update.
    '''

    def __init__(self, actor, soft_update_factor):
        '''
        Initializes the actor target model and loads the pretrained model if need be.

        Arguments:
        `actor`: actor network object
        `soft_update_factor`: factor for actor weights used for soft update. aka tau.
        '''

        # Initialize variables
        self.actor = actor
        self.actor.set_target(self)
        self.soft_update_factor = soft_update_factor

        # Initialize actor target network
        self.model = self.initialize_model()
        rospy.loginfo(f'Initialized actor target: {self.model.summary()}')
        self.model_filepath = self.actor.model_filepath
        self.model_name = "actor_target"
        if self.actor.load_pretrained_model:
            model_path = self.model_filepath + self.model_name + PRETRAINED_MODEL_VERSION + ".h5"
            self.model.load_weights(model_path)
            rospy.loginfo(f'Loaded actor target from file: {self.model.summary()}')


    def initialize_model(self):
        '''
        Initialize target neural network model using the initialization function of actor

        Returns:
        `model`: target neural network model
        '''

        model = self.actor.initialize_model()
        model.set_weights(self.actor.model.get_weights())

        rospy.logdebug(f'Weights actor: {self.actor.model.get_weights()}')

        return model


    def get_action_batch(self, state_batch):
        '''
        Calculates the actions of a batch of states using the actor target.

        Arguments:
        `state_batch`: batch of states

        Returns:
        `action_batch`: batch of actions calculated by the actor
        '''

        action_batch = self.model(state_batch, training=True)

        return action_batch


    def update_weights(self):
        '''
        Updates the weights of the actor target using soft update.
        '''

        rospy.logdebug("Updating actor target weights...")
        tau = self.soft_update_factor

        rospy.logdebug(f'Actor weights before actor_target update: {self.actor.model.variables}')
        rospy.logdebug(f'Actor target weights before actor_target update: {self.model.variables}')

        for (a, b) in zip(self.model.variables, self.actor.model.variables):
            a.assign(b * tau + a * (1 - tau))

        rospy.logdebug(f'Actor weights after actor_target update: {self.actor.model.variables}')
        rospy.logdebug(f'Actor target after actor_target update: {self.model.variables}')


class Critic():
    '''
    The critic network that determines expected future reward of a state and an action. It receives batches to train the network.
    '''

    def __init__(self, learning_rate, model_filepath, load_pretrained_model):
        '''
        Initializes the critic network with optimizer and loads pretrained model from file if need be.

        Arguments:
        `learning_rate`: rate at which the weights are adjusted by
        `model_filepath`: path where to save an load the model
        `load_pretrained_model`: whether to load an allready existing model
        '''

        # Initialize variables
        self.learning_rate = learning_rate

        # Initialize critic model
        self.model = self.initialize_model()
        rospy.loginfo(f'Initialized critic: {self.model.summary()}')
        self.model_filepath = model_filepath
        self.model_name = "critic"
        self.load_pretrained_model = load_pretrained_model
        if self.load_pretrained_model:

            model_path = self.model_filepath + self.model_name + PRETRAINED_MODEL_VERSION + ".h5"
            self.model.load_weights(model_path)
            rospy.loginfo(f'Loaded critic from file: {self.model.summary()}')

        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)


    def initialize_model(self):
        '''
        Initializes the critic neural network model.

        Returns:
        `model`: neural network model of critic
        '''

        # Input network state without images
        input_layer_state_rest= layers.Input(shape=(NUMBER_OF_STATE_VALUES_WITHOUT_IMAGES))
        dense_layer_state_rest = layers.Dense(64, activation="relu")(input_layer_state_rest)
        reshape_layer_state_rest = layers.Reshape((1, 1, 64))(dense_layer_state_rest)

        # Input network images
        input_layer_state_image_1 = layers.Input(shape=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1))
        input_layer_state_image_2 = layers.Input(shape=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1))
        input_layer_state_image_3 = layers.Input(shape=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1))
        concat_input_images_layer = layers.Concatenate()([
            input_layer_state_image_1,
            input_layer_state_image_2,
            input_layer_state_image_3
        ])
        convolution_layer_images_1 = Conv2D(64, 6, 2, activation='relu')(concat_input_images_layer)
        max_pool_layer_images_1 = MaxPool2D(3)(convolution_layer_images_1)
        convolution_layer_images_2 = Conv2D(64, 5, 1, activation='relu')(max_pool_layer_images_1)
        convolution_layer_images_3 = Conv2D(64, 5, 1, activation='relu')(convolution_layer_images_2)
        convolution_layer_images_4 = Conv2D(64, 5, 1, activation='relu')(convolution_layer_images_3)
        convolution_layer_images_5 = Conv2D(64, 5, 1, activation='relu')(convolution_layer_images_4)
        convolution_layer_images_6 = Conv2D(64, 5, 1, activation='relu')(convolution_layer_images_5)
        convolution_layer_images_7 = Conv2D(64, 5, 1, activation='relu')(convolution_layer_images_6)
        max_pool_layer_images_3 = MaxPool2D(3)(convolution_layer_images_7)
        convolution_layer_1 = Conv2D(64, 3, 1, activation='relu')(max_pool_layer_images_3)
        convolution_layer_2 = Conv2D(64, 3, 1, activation='relu')(convolution_layer_1)
        convolution_layer_3 = Conv2D(64, 3, 1, activation='relu')(convolution_layer_2)
        convolution_layer_4 = Conv2D(64, 3, 1, activation='relu')(convolution_layer_3)
        max_pool_layer_1 = layers.MaxPool2D(2)(convolution_layer_4)
        convolution_layer_7 = Conv2D(64, 3, 1, activation='relu')(max_pool_layer_1)
        convolution_layer_8 = Conv2D(64, 3, 1, activation='relu')(convolution_layer_7)

        # Input network action
        input_layer_action = layers.Input(shape=(NUMBER_OF_ACTION_VALUES))
        dense_layer_action_1 = layers.Dense(256, activation="relu")(input_layer_action)
        dense_layer_action_2 = layers.Dense(64, activation="relu")(dense_layer_action_1)
        reshape_layer_action= layers.Reshape((1, 1, 64))(dense_layer_action_2)

        # Concat inputs
        concat_input_layer = layers.Concatenate(axis=1)(
            [
                reshape_layer_state_rest,
                reshape_layer_action,
                convolution_layer_8
            ]
        )
        dense_layer_1 = layers.Dense(64, activation="relu")(concat_input_layer)
        flatten_layer_1 = Flatten()(dense_layer_1)
        dense_layer_2 = layers.Dense(64, activation="relu")(flatten_layer_1)
        output_layer = layers.Dense(1)(dense_layer_2)

        # Create model
        model = tf.keras.Model(
            [
                input_layer_state_rest,
                input_layer_state_image_1,
                input_layer_state_image_2,
                input_layer_state_image_3,
                input_layer_action
            ],
                output_layer
        )

        return model


    def set_target(self, target):
        '''
        Sets the critic target to make it accessible during training step.

        Arguments:
        `critic_target`: critic target object
        '''
        self.target = target


    def get_value_batch(self, state_batch, action_batch):
        '''
        Calculates the expected future return of a batch of states and actions using the critic.

        Arguments:
        `state_batch`: batch of states
        `action_batch`: batch of actions

        Returns:
        `value_batch`: batch of expected future return values calculated by the critic
        '''

        # Create model input
        state_batch.append(action_batch)
        model_input = state_batch

        value_batch = self.model(model_input, training=True)
        rospy.logdebug(f'Value batch: {value_batch}')

        return value_batch


    def train(self, batch):
        '''
        Update critic weights by calculating and applying the gradient of the critic error.

        Argmuents:
        `state_batch_tensor`: batch of state tensors
        `action_batch_tensor`: batch of action tensors
        `reward_batch_tensor`: batch of reward tensors
        `next_state_batch_tensor`: batch of next_state tensors
        `not_done_batch_tensor`: batch of not_done tensors. true, if next_state is not a terminal state

        Returns:
        `critic_error`: critic error of batch
        '''

        rospy.logdebug("Updating critic.")
 
        state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = self.actor.get_batch_tensors(batch) # TODO Pass as function args, works as before?

        rospy.logdebug(f'Reward batch: {reward_batch}')

        with tf.GradientTape() as tape:
            # a(s') = a'
            actor_target_actions = self.actor.target.get_action_batch(next_state_batch)
            rospy.logdebug(f'Actor target actions: {actor_target_actions.shape}: {actor_target_actions} {actor_target_actions.dtype}')
            # Q(s', a(s')) / estimated future reward in s' with a
            critic_target_values = self.target.get_value_batch(next_state_batch, actor_target_actions)
            rospy.logdebug(f'Critic target values {critic_target_values.shape}: {critic_target_values} {critic_target_values.dtype}')
            # Q_T(s, a)/q_target = reward + disount_factor * future_reward(s',a(s'))
            future_reward_target = tf.add(reward_batch, tf.multiply(tf.multiply(not_done_batch, REWARD_DISCOUNT_FACTOR), critic_target_values))
            rospy.logdebug(f'Future reward target: {future_reward_target}')
            # q_prediction = Q(s,a) / critic_values
            future_reward_prediction = self.get_value_batch(state_batch, action_batch)
            rospy.logdebug(f'Future reward prediciton values {future_reward_prediction.shape}: {future_reward_prediction} {future_reward_prediction.dtype}')
            # critic_error = 1/n (Q(s,a) - Q_T(s,a))² / MSE of critic network
            critic_error =  tf.math.reduce_mean(tf.math.square(future_reward_target - future_reward_prediction))
            rospy.logdebug(f'Critic error {critic_error.shape}: {critic_error} {critic_error.dtype}')

        # calculate gradiant(derivative) of critic_error-function
        critic_error_gradient = tape.gradient(critic_error, self.model.trainable_variables)
        rospy.logdebug(f'Critic error gradient: {critic_error_gradient}')

        # Change weights(trainable variables) in direction described by gradient
        rospy.logdebug(f'Critic weights before critic update: {self.model.trainable_variables}')
        rospy.logdebug(f'Critic target weights before critic update: {self.target.model.trainable_variables}')
        self.optimizer.apply_gradients(
            zip(critic_error_gradient, self.model.trainable_variables)
        )
        rospy.logdebug(f'Critic weights after critic update: {self.model.trainable_variables}')
        rospy.logdebug(f'Critic target weights after critic update : {self.target.model.trainable_variables}')

        test = False
        if test:
            state_batch.pop() # gradient tape seems to add an element at the end of state_batch
            # q_prediction = Q(s,a) / critic_values
            future_reward_prediction_test = self.get_value_batch(state_batch, action_batch)
            # critic_error = 1/n (Q(s,a) - Q_T(s,a))² / MSE of critic network
            critic_error_test =  tf.math.reduce_mean(tf.math.square(future_reward_target - future_reward_prediction_test))
            rospy.logdebug(f'Critic error {critic_error.shape}: {critic_error} {critic_error.dtype}')
            rospy.logdebug(f'Critic error test (<critc_error?) {critic_error_test.shape}: {critic_error_test} {critic_error_test.dtype}')
            rospy.loginfo(f'Critic update successful(lowered critic error)?: {critic_error_test < critic_error}, difference={critic_error_test-critic_error}')
        rospy.logdebug(f"Finished critic update with gradient: {critic_error_gradient}")

        return critic_error


    def set_actor(self, actor):
        '''
        Sets the actor to make it accessible during training step

        Arguments:
        `actor`: actor object
        '''

        self.actor = actor



class CriticTarget():
    '''
    The critic target network that is a copy of the critic network, which is updated with a soft update.
    '''

    def __init__(self, critic, soft_update_factor):
        '''
        Initializes the critic target model and loads the pretrained model if need be.

        Arguments:
        `actor`: actor network object
        `soft_update_factor`: factor for actor weights used for soft update. aka tau.
        '''

        # Initialize varibles
        self.critic = critic
        self.critic.set_target(self)
        self.soft_update_factor = soft_update_factor

        # Initialize critic model
        self.model = self.initialize_model()
        rospy.loginfo(f'Initialized critic target: {self.model.summary()}')
        self.model_filepath = self.critic.model_filepath
        self.model_name = "critic_target"
        self.load_pretrained_model = self.critic.load_pretrained_model
        if self.load_pretrained_model:
            model_path = self.model_filepath + self.model_name + PRETRAINED_MODEL_VERSION + ".h5"
            self.model.load_weights(model_path)
            rospy.loginfo(f'Loaded critic target from file: {self.model.summary()}')


    def initialize_model(self):
        '''
        Initialize target neural network model using the initialization function of critic

        Returns:
        `model`: target neural network model
        '''

        model = self.critic.initialize_model()
        model.set_weights(self.critic.model.get_weights())

        rospy.logdebug(f'Weights critic: {self.critic.model.get_weights()}')
        rospy.logdebug(f'Weights critic target: {model.get_weights()}')

        return model


    def update_weights(self):
        '''
        Updates the weights of the critic target using soft update.
        '''

        rospy.logdebug("Updating critic target weights...")
        tau = self.soft_update_factor

        rospy.logdebug(f'Critic weights before critic target update: {self.critic.model.variables}')
        rospy.logdebug(f'Critic target weights before critic target weights: {self.model.variables}')

        for (a, b) in zip(self.model.variables, self.critic.model.variables):
            a.assign(b * tau + a * (1 - tau))

        rospy.logdebug(f'Critic weights after critic target update: {self.critic.model.variables}')
        rospy.logdebug(f'Critic target weights after critic target update: {self.model.variables}')


    def get_value_batch(self, state_batch, action_batch):
        '''
        Calculates the expected future return of a batch of states and actions using the critic target.

        Arguments:
        `state_batch`: batch of states
        `action_batch`: batch of actions

        Returns:
        `value_batch`: batch of expected future return values calculated by the critic target
        '''

        # Create model input
        state_batch.append(action_batch)
        model_input = state_batch

        value_batch = self.model(model_input, training=True)
        rospy.logdebug(f'Value batch: {value_batch}')
        return value_batch



def main():
    '''
    Initialize agent_model node by creating actor, critic and their target networks at training and creating only the actor at evaluation.
    '''

    rospy.init_node('agent_model')

    rospy.loginfo(f"Hardware acceleration {'enabled' if len(tf.config.list_physical_devices('GPU'))>0 else 'not enabled'}.")

    if ONLY_ACTOR:

        actor = Actor(
            ACTOR_LEARNING_RATE,
            ACTION_MAXIMUM,
            MODEL_FILEPATH,
            LOAD_PRETRAINED_MODEL,
            PRETRAINED_MODEL_VERSION,
            NUMBER_OF_STATE_VALUES_WITHOUT_IMAGES,
            INPUT_IMAGE_HEIGHT,
            INPUT_IMAGE_WIDTH,
            NUMBER_OF_ACTION_VALUES,
            VERSION_NAME,
            WORKSPACE_LOWER_BOUND_X,
            WORKSPACE_UPPER_BOUND_X,
            WORKSPACE_LOWER_BOUND_Y,
            WORKSPACE_UPPER_BOUND_Y,
            WORKSPACE_LOWER_BOUND_Z,
            WORKSPACE_UPPER_BOUND_Z,
            GRIPPER_WIDTH_OPEN
        )

    else:

        actor = Actor(
            ACTOR_LEARNING_RATE,
            ACTION_MAXIMUM,
            MODEL_FILEPATH,
            LOAD_PRETRAINED_MODEL,
            PRETRAINED_MODEL_VERSION,
            NUMBER_OF_STATE_VALUES_WITHOUT_IMAGES,
            INPUT_IMAGE_HEIGHT,
            INPUT_IMAGE_WIDTH,
            NUMBER_OF_ACTION_VALUES,
            VERSION_NAME,
            WORKSPACE_LOWER_BOUND_X,
            WORKSPACE_UPPER_BOUND_X,
            WORKSPACE_LOWER_BOUND_Y,
            WORKSPACE_UPPER_BOUND_Y,
            WORKSPACE_LOWER_BOUND_Z,
            WORKSPACE_UPPER_BOUND_Z,
            GRIPPER_WIDTH_OPEN
        )
        actor_target = Actor_Target(actor, SOFT_UPDATE_FACTOR)

        critic = Critic(CRITIC_LEARNING_RATE, MODEL_FILEPATH, LOAD_PRETRAINED_MODEL)
        critic_target = CriticTarget(critic, SOFT_UPDATE_FACTOR)

        actor.set_critic(critic)
        critic.set_actor(actor)

    rate = rospy.Rate(10)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
