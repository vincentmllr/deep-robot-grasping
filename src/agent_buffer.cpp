#include "ros/ros.h"
#include "panda_deep_grasping/Sample.h"
#include "panda_deep_grasping/GetBatch.h"
#include "geometry_msgs/Point.h"
#include <stdlib.h>     // srand, rand (set seed with: srand(time(NULL));)
#include <time.h>
#include <random>      // for more advanced random distriubutions

#include <fstream>
#include <string>
// include headers that implement a archive in simple text format
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/tmpdir.hpp>
#include <boost/archive/archive_exception.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/assume_abstract.hpp>


// Constants
const int BATCH_SIZE = 16; 
const int MINIMUM_REPLAY_BUFFER_SIZE = 100;
const int MAXIMUM_REPLAY_BUFFER_SIZE = 1000000; // Should be 10^6 // Alternative wording: REPLAY_BUFFER_CAPACITY?
const int RANDOM_SEED = time(NULL);
std::default_random_engine RANDOM_GENERATOR(RANDOM_SEED);
const float REWARD_SUCCESS = 1; // QTOpt 1 // Before +5
const float REWARD_FAILED = 0; // QTOpt 0 // Before -5
const float REWARD_CONSTANT = -0.025; // QTOpt 0.05 // Before -0.1
const float REWARD_HEIGHT = 0.05; // factor for height([0,0.38]),should be lower than reward_success, if received with max height value (0.38) over all timesteps(40) // 0.38*40*0.005 = 0.76 < 1 !
const float REWARD_ERROR = 0; // Before -10
const int REWARD_CODE_BASIC = 0;
const int REWARD_CODE_SUCCESS = 1;
const int REWARD_CODE_FAILED = 2;
const int REWARD_CODE_ERROR = 3;
const float INITIAL_POSITION_Z = 0.5;
const int SAVE_FREQUENCY = 50;

// Variables to set manually
const std::string FILE_PATH = "/../panda_deep_grasping/simulation_ws/src/panda-deep-grasping/replay_buffer_files";
const bool load_buffer = false;
const std::string load_buffer_version = "2.0";
const std::string save_buffer_version = "2.0";


/**
 * @brief Class to define a serializable Sample object containing variables of the state, aciton, reward and next_state.
 * 
 */
class Sample {

    private:

        friend class boost::serialization::access;
        template<class Archive>

        /**
         * @brief Declare variables to be serialized.
         * 
         * @param ar 
         * @param version 
         */
        void serialize(Archive & ar, const unsigned int version) {
            
            ar & state_endeffector_position_x;
            ar & state_endeffector_position_y;
            ar & state_endeffector_position_z;
            ar & state_joint_angle_1;
            ar & state_joint_angle_2;
            ar & state_joint_angle_3;
            ar & state_joint_angle_4;
            ar & state_joint_angle_5;
            ar & state_joint_angle_6;
            ar & state_joint_angle_7;
            ar & state_gripper_width;
            ar & state_filepath_1;
            ar & state_filepath_2;
            ar & state_filepath_3;
            ar & action_dx;
            ar & action_dy;
            ar & action_dz;
            ar & action_droll;
            ar & action_dpitch;
            ar & action_dyaw;
            ar & action_grasp;
            ar & reward;
            ar & next_state_endeffector_position_x;
            ar & next_state_endeffector_position_y;
            ar & next_state_endeffector_position_z;
            ar & next_state_joint_angle_1;
            ar & next_state_joint_angle_2;
            ar & next_state_joint_angle_3;
            ar & next_state_joint_angle_4;
            ar & next_state_joint_angle_5;
            ar & next_state_joint_angle_6;
            ar & next_state_joint_angle_7;
            ar & next_state_gripper_width;
            ar & next_state_filepath_1;
            ar & next_state_filepath_2;
            ar & next_state_filepath_3;

        }

        double state_endeffector_position_x;
        double state_endeffector_position_y;
        double state_endeffector_position_z;
        double state_joint_angle_1;
        double state_joint_angle_2;
        double state_joint_angle_3;
        double state_joint_angle_4;
        double state_joint_angle_5;
        double state_joint_angle_6;
        double state_joint_angle_7;
        double state_gripper_width;
        std::string state_filepath_1;
        std::string state_filepath_2;
        std::string state_filepath_3;
        float action_dx;
        float action_dy;
        float action_dz;
        float action_droll;
        float action_dpitch;
        float action_dyaw;
        int action_grasp;
        float reward;
        double next_state_endeffector_position_x;
        double next_state_endeffector_position_y;
        double next_state_endeffector_position_z;
        double next_state_joint_angle_1;
        double next_state_joint_angle_2;
        double next_state_joint_angle_3;
        double next_state_joint_angle_4;
        double next_state_joint_angle_5;
        double next_state_joint_angle_6;
        double next_state_joint_angle_7;
        double next_state_gripper_width;
        std::string next_state_filepath_1;
        std::string next_state_filepath_2;
        std::string next_state_filepath_3;

    public:
        
        /**
         * @brief Construct a new Sample object
         * 
         */
        Sample(){
        };

        /**
         * @brief Construct a new Sample object and setting variables
         * 
         */
        Sample(
            double _state_endeffector_position_x,
            double _state_endeffector_position_y,
            double _state_endeffector_position_z,
            double _state_joint_angle_1,
            double _state_joint_angle_2,
            double _state_joint_angle_3,
            double _state_joint_angle_4,
            double _state_joint_angle_5,
            double _state_joint_angle_6,
            double _state_joint_angle_7,
            double _state_gripper_width,
            std::string _state_filepath_1,
            std::string _state_filepath_2,
            std::string _state_filepath_3,
            float _action_dx,
            float _action_dy,
            float _action_dz,
            float _action_droll,
            float _action_dpitch,
            float _action_dyaw,
            int _action_grasp,
            float _reward,
            double _next_state_endeffector_position_x,
            double _next_state_endeffector_position_y,
            double _next_state_endeffector_position_z,
            double _next_state_joint_angle_1,
            double _next_state_joint_angle_2,
            double _next_state_joint_angle_3,
            double _next_state_joint_angle_4,
            double _next_state_joint_angle_5,
            double _next_state_joint_angle_6,
            double _next_state_joint_angle_7,
            double _next_state_gripper_width,
            std::string _next_state_filepath_1,
            std::string _next_state_filepath_2,
            std::string _next_state_filepath_3
            ) :
            state_endeffector_position_x(_state_endeffector_position_x),
            state_endeffector_position_y(_state_endeffector_position_y),
            state_endeffector_position_z(_state_endeffector_position_z),
            state_joint_angle_1(_state_joint_angle_1),
            state_joint_angle_2(_state_joint_angle_2),
            state_joint_angle_3(_state_joint_angle_3),
            state_joint_angle_4(_state_joint_angle_4),
            state_joint_angle_5(_state_joint_angle_5),
            state_joint_angle_6(_state_joint_angle_6),
            state_joint_angle_7(_state_joint_angle_7),
            state_gripper_width(_state_gripper_width),
            state_filepath_1(_state_filepath_1),
            state_filepath_2(_state_filepath_2),
            state_filepath_3(_state_filepath_3),
            action_dx(_action_dx),
            action_dy(_action_dy),
            action_dz(_action_dz),
            action_droll(_action_droll),
            action_dpitch(_action_dpitch),
            action_dyaw(_action_dy),
            action_grasp(_action_grasp),
            reward(_reward),
            next_state_endeffector_position_x(_next_state_endeffector_position_x),
            next_state_endeffector_position_y(_next_state_endeffector_position_y),
            next_state_endeffector_position_z(_next_state_endeffector_position_z),
            next_state_joint_angle_1(_next_state_joint_angle_1),
            next_state_joint_angle_2(_next_state_joint_angle_2),
            next_state_joint_angle_3(_next_state_joint_angle_3),
            next_state_joint_angle_4(_next_state_joint_angle_4),
            next_state_joint_angle_5(_next_state_joint_angle_5),
            next_state_joint_angle_6(_next_state_joint_angle_6),
            next_state_joint_angle_7(_next_state_joint_angle_7),
            next_state_gripper_width(_next_state_gripper_width),
            next_state_filepath_1(_next_state_filepath_1),
            next_state_filepath_2(_next_state_filepath_2),
            next_state_filepath_3(_next_state_filepath_3) {
        }

        /**
         * @brief Get the state endeffector position.
         * 
         * @return geometry_msgs::Point cartesian endeffector position
         */
        geometry_msgs::Point get_state_endeffector_position() {
            
            geometry_msgs::Point state_endeffector_position;
            state_endeffector_position.x = state_endeffector_position_x;
            state_endeffector_position.y = state_endeffector_position_y;
            state_endeffector_position.z = state_endeffector_position_z;
            return state_endeffector_position;

        }

        /**
         * @brief Get the state joint angles.
         * 
         * @return std::vector<double> joint angles of state
         */
        std::vector<double> get_state_joint_angles() {
            std::vector<double> state_joint_angles = {
                state_joint_angle_1,
                state_joint_angle_2,
                state_joint_angle_3,
                state_joint_angle_4,
                state_joint_angle_5,
                state_joint_angle_6,
                state_joint_angle_7
            };
            return state_joint_angles;
        }

        /**
         * @brief Get the state gripper width.
         * 
         * @return double gripper width of state
         */
        double get_state_gripper_width() {
            return state_gripper_width;
        }

        /**
         * @brief Get the state filepaths.
         * 
         * @return std::vector<std::string> state filepaths
         */
        std::vector<std::string> get_state_filepaths() {
            std::vector<std::string> state_filepaths = {state_filepath_1, state_filepath_2, state_filepath_3};
            return state_filepaths;
        }

        /**
         * @brief Get the relative movement of the action.
         * 
         * @return std::vector<float> relative movement of action
         */
        std::vector<float> get_action_relative_movement() {
            std::vector<float> action_relative_movement = {action_dx, action_dy, action_dz, action_droll, action_dpitch, action_dyaw};
            return action_relative_movement;
        }

        /**
         * @brief Get the grasp command of action.
         * 
         * @return int  grasp command of action
         */
        int get_action_grasp() {
            return action_grasp;
        }

        /**
         * @brief Get the reward as code.
         * 
         * @return float reward code
         */
        float get_reward() {
            return reward;
        }

        /**
         * @brief Get the endeffector position of the next state.
         * 
         * @return geometry_msgs::Point endeffector position of next state
         */
        geometry_msgs::Point get_next_state_endeffector_position() {
            
            geometry_msgs::Point next_state_endeffector_position;
            next_state_endeffector_position.x = next_state_endeffector_position_x;
            next_state_endeffector_position.y = next_state_endeffector_position_y;
            next_state_endeffector_position.z = next_state_endeffector_position_z;
            return next_state_endeffector_position;

        }

        /**
         * @brief Get the joint angles of the next state.
         * 
         * @return std::vector<double> joint angles of next state
         */
        std::vector<double> get_next_state_joint_angles() {
            std::vector<double> next_state_joint_angles = {
                next_state_joint_angle_1,
                next_state_joint_angle_2,
                next_state_joint_angle_3,
                next_state_joint_angle_4,
                next_state_joint_angle_5,
                next_state_joint_angle_6,
                next_state_joint_angle_7
            };
            return next_state_joint_angles;
        }

        /**
         * @brief Get the gripper width of the next state.
         * 
         * @return double gripper width of next state 
         */
        double get_next_state_gripper_width() {
            return next_state_gripper_width;
        }

        /**
         * @brief Get the filepaths of the next state.
         * 
         * @return std::vector<std::string> filepaths of next state 
         */
        std::vector<std::string> get_next_state_filepaths() {
            std::vector<std::string> next_state_filepaths = {next_state_filepath_1, next_state_filepath_2, next_state_filepath_3};
            return next_state_filepaths;
        }

};

/**
 * @brief Class to define a serializable Replay_Buffer object filled with Sample objects.
 * 
 */
class Replay_Buffer {
    
    friend class boost::serialization::access;
    int last_empty_index;
    int save_counter;
    bool buffer_at_max_cap;
    std::array<Sample, MAXIMUM_REPLAY_BUFFER_SIZE> replay_buffer_boost;
    template<class Archive>

    /**
     * @brief Declare variables to be serialized.
     * 
     * @param ar 
     * @param version 
     */
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & replay_buffer_boost;
        ar & last_empty_index;
    }

public:

    /**
     * @brief Construct a new Replay_Buffer object
     * 
     */
    Replay_Buffer(){
        last_empty_index = 0;
        save_counter = 0;
        buffer_at_max_cap = false;
    }

    /**
     * @brief Add Sample to the replay buffer.
     * 
     * @param sample Sample object to add to replay buffer
     */
    void append(Sample sample){

        if(last_empty_index == MAXIMUM_REPLAY_BUFFER_SIZE-1) {
            ROS_INFO("Replay buffer full: Starting to replace old samples.");
	    buffer_at_max_cap = true;
            last_empty_index = 0;
        }

        ROS_DEBUG("Append sample to replay buffer at index %d.", last_empty_index);
        replay_buffer_boost[last_empty_index++] = sample;

    }

    /**
     * @brief Get Sample object at specific index in replay buffer.
     * 
     * @param index location of sample in replay buffer
     * @return Sample sample at given index
     */
    Sample at(int index) {
        ROS_DEBUG("Retrieving sample from replay buffer at index %d.", index);
        return replay_buffer_boost.at(index);
    }

    /**
     * @brief Retrieve a counter that determines when to save the replay buffer and add increment to counter.
     * 
     * @return int current counter value
     */
    int get_save_counter() {
        return save_counter++;
    }

    /**
     * @brief Get the capacity of the replay buffer
     * 
     * @return int capacity of replay buffer
     */
    int get_size() {
        return replay_buffer_boost.size(); // TODO Differentiate betweeen get_size and get_current_size(): Rename get_size() to get_capacity?
    }

    /**
     * @brief Get the current size of the replay buffer
     * 
     * @return int size of replay buffer
     */
    int get_current_size() {
        return last_empty_index;
    }

    /**
     * @brief Retrive whether replay buffer is at maximum capacity.
     * 
     * @return true if replay buffer is at maximum capacity
     */
    bool get_buffer_at_max_cap() {
	return buffer_at_max_cap;
    }

};

// Create replay buffer object
Replay_Buffer replay_buffer;

/**
 * @brief Transform a sample message to a sample object and add it to the replay buffer.
 * 
 * @param sample sample message
 */
void add_sample(panda_deep_grasping::Sample sample) {

    std::vector<std::string> state_image_filepaths = sample.state.image_filepaths;
    std::vector<std::string> next_state_image_filepaths = sample.next_state.image_filepaths;
    ROS_DEBUG("Received sample with %ld images in each state", state_image_filepaths.size());
    ROS_DEBUG("Adding sample w/ state %f %f %f %f %f %f %f %f %f %f %f %s %s %s",
        sample.state.endeffector_position.x,
        sample.state.endeffector_position.y,
        sample.state.endeffector_position.z,
        sample.state.joint_angles[0],
        sample.state.joint_angles[1],
        sample.state.joint_angles[2],
        sample.state.joint_angles[3],
        sample.state.joint_angles[4],
        sample.state.joint_angles[5],
        sample.state.joint_angles[6],
        sample.state.gripper_width,
        sample.state.image_filepaths.at(0).c_str(),
        sample.state.image_filepaths.at(1).c_str(),
        sample.state.image_filepaths.at(2).c_str()
    );

    Sample new_sample(
        sample.state.endeffector_position.x,
        sample.state.endeffector_position.y,
        sample.state.endeffector_position.z,
        sample.state.joint_angles[0],
        sample.state.joint_angles[1],
        sample.state.joint_angles[2],
        sample.state.joint_angles[3],
        sample.state.joint_angles[4],
        sample.state.joint_angles[5],
        sample.state.joint_angles[6],
        sample.state.gripper_width,
        sample.state.image_filepaths.at(0),
        sample.state.image_filepaths.at(1),
        sample.state.image_filepaths.at(2),
        sample.action.relative_movement[0],
        sample.action.relative_movement[1],
        sample.action.relative_movement[2],
        sample.action.relative_movement[3],
        sample.action.relative_movement[4],
        sample.action.relative_movement[5],
        sample.action.grasp,
        sample.reward.reward,
        sample.next_state.endeffector_position.x,
        sample.next_state.endeffector_position.y,
        sample.next_state.endeffector_position.z,
        sample.next_state.joint_angles[0],
        sample.next_state.joint_angles[1],
        sample.next_state.joint_angles[2],
        sample.next_state.joint_angles[3],
        sample.next_state.joint_angles[4],
        sample.next_state.joint_angles[5],
        sample.next_state.joint_angles[6],
        sample.next_state.gripper_width,
        sample.next_state.image_filepaths.at(0),
        sample.next_state.image_filepaths.at(1),
        sample.next_state.image_filepaths.at(2)
    );

    replay_buffer.append(new_sample);

}

/**
 * @brief Callback function for add_sample-topic that adds a sample to the replay buffer
 * 
 * @param msg Sample containing state, action, next_state, reward
 */
void add_sample_callback(const panda_deep_grasping::Sample::ConstPtr& msg) {
    
    panda_deep_grasping::Sample new_sample;
    new_sample.state = msg->state;
    new_sample.action = msg->action;
    new_sample.reward = msg->reward;
    new_sample.next_state = msg->next_state;

    add_sample(new_sample);
    ROS_DEBUG("Added sample to replay buffer.");

    bool saved = false;
    if(replay_buffer.get_save_counter() % SAVE_FREQUENCY == 0) {
        ROS_DEBUG("Saving replay buffer to file.");
        std::string filename(boost::archive::tmpdir());
        filename += FILE_PATH;
        filename += "/replay_buffer" + save_buffer_version +".txt";

        // create and open a character archive for output
        std::ofstream outfilestream(filename.c_str());

        // Save data to archive
        {
            boost::archive::text_oarchive outarchive(outfilestream);
            // Write instance to archive
            outarchive << replay_buffer;
        }
        ROS_DEBUG("Saved replay buffer to file.");
        saved =true;
    }

    ROS_INFO("agent_buffer/add_sample: Added sample @%d w/ z_ee^s=%f, dz=%f, g=%d, z_ee^s'=%f, r=%f%s",
        replay_buffer.get_current_size()-1,
        new_sample.state.endeffector_position.z,
        new_sample.action.relative_movement.at(2),
        new_sample.action.grasp,
        new_sample.next_state.endeffector_position.z,
        new_sample.reward.reward,
        saved ? ", saved replay buffer" : ""
    );

}

/**
 * @brief Get the sample object at given index as sample message.
 * 
 * @param index location of sample
 * @return panda_deep_grasping::Sample sample message
 */
panda_deep_grasping::Sample get_sample(int index) {

    Sample replay_buffer_sample = replay_buffer.at(index);

    panda_deep_grasping::Sample sample;

    sample.state.endeffector_position = replay_buffer_sample.get_state_endeffector_position();
    std::vector<double> state_joint_angles = replay_buffer_sample.get_state_joint_angles();
    sample.state.joint_angles[0] = state_joint_angles.at(0);
    sample.state.joint_angles[1] = state_joint_angles.at(1);
    sample.state.joint_angles[2] = state_joint_angles.at(2);
    sample.state.joint_angles[3] = state_joint_angles.at(3);
    sample.state.joint_angles[4] = state_joint_angles.at(4);
    sample.state.joint_angles[5] = state_joint_angles.at(5);
    sample.state.joint_angles[6] = state_joint_angles.at(6);
    sample.state.gripper_width = replay_buffer_sample.get_state_gripper_width();
    std::vector<std::string> state_image_filepaths = replay_buffer_sample.get_state_filepaths();
    ROS_DEBUG("Received sample with %ld images in each state", state_image_filepaths.size());
    sample.state.image_filepaths.push_back(state_image_filepaths.at(0));
    sample.state.image_filepaths.push_back(state_image_filepaths.at(1));
    sample.state.image_filepaths.push_back(state_image_filepaths.at(2));
        
    std::vector<float> action_relative_movement = replay_buffer_sample.get_action_relative_movement();
    ROS_DEBUG("Received sample with %ld movement variables in action", action_relative_movement.size());
    sample.action.relative_movement[0] = action_relative_movement.at(0);
    sample.action.relative_movement[1] = action_relative_movement.at(1);
    sample.action.relative_movement[2] = action_relative_movement.at(2);
    sample.action.relative_movement[3] = action_relative_movement.at(3);
    sample.action.relative_movement[4] = action_relative_movement.at(4);
    sample.action.relative_movement[5] = action_relative_movement.at(5);
    sample.action.grasp = replay_buffer_sample.get_action_grasp();

    // Transform reward code to reward value
    float reward = 0;
    float reward_code = replay_buffer_sample.get_reward();
    switch ((int)reward_code)
    {
    case REWARD_CODE_BASIC:
        reward = REWARD_HEIGHT * (INITIAL_POSITION_Z - replay_buffer_sample.get_next_state_endeffector_position().z); // the lower the better
        break;
    case REWARD_CODE_SUCCESS:
        reward = REWARD_SUCCESS;
        break;
    case REWARD_CODE_FAILED:
        reward = REWARD_FAILED;
        break;
    case REWARD_CODE_ERROR:
        reward = REWARD_ERROR;
        break;     
    default:
        break;
    }
    sample.reward.reward = reward;

    sample.next_state.endeffector_position = replay_buffer_sample.get_next_state_endeffector_position();
    std::vector<double> next_state_joint_angles = replay_buffer_sample.get_next_state_joint_angles();
    sample.next_state.joint_angles[0]=next_state_joint_angles.at(0);
    sample.next_state.joint_angles[1]=next_state_joint_angles.at(1);
    sample.next_state.joint_angles[2]=next_state_joint_angles.at(2);
    sample.next_state.joint_angles[3]=next_state_joint_angles.at(3);
    sample.next_state.joint_angles[4]=next_state_joint_angles.at(4);
    sample.next_state.joint_angles[5]=next_state_joint_angles.at(5);
    sample.next_state.joint_angles[6]=next_state_joint_angles.at(6);
    sample.next_state.gripper_width = replay_buffer_sample.get_next_state_gripper_width();
    std::vector<std::string> next_state_image_filepaths = replay_buffer_sample.get_next_state_filepaths();
    sample.next_state.image_filepaths.push_back(next_state_image_filepaths.at(0));
    sample.next_state.image_filepaths.push_back(next_state_image_filepaths.at(1));
    sample.next_state.image_filepaths.push_back(next_state_image_filepaths.at(2));

    return sample;

}

/**
 * @brief Create a batch of samples for training.
 * 
 * @param req empty
 * @param res batch of training samples
 * 
 * @return true if creation of batch successful
 */
bool getBatch(panda_deep_grasping::GetBatch::Request& req, panda_deep_grasping::GetBatch::Response& res){

    std::vector<panda_deep_grasping::Sample> batch;
    std::vector<int> batch_indices;

    if(replay_buffer.get_current_size() >= MINIMUM_REPLAY_BUFFER_SIZE) {
 
        std::uniform_int_distribution<int> batch_distribution(0, (replay_buffer.get_current_size()-1));

        for(int i=0; i<BATCH_SIZE; i++) {

            int sample_index = batch_distribution(RANDOM_GENERATOR);
            ROS_DEBUG("Getting sample at index %d.", sample_index);
            panda_deep_grasping::Sample sample = get_sample(sample_index);
            batch.push_back(sample);
            batch_indices.push_back(sample_index);

        }
        
        ROS_DEBUG("Created batch of %ld samples.", batch.size());
        res.batch = batch;

        if(batch.size() >= 0) {
            ROS_INFO("agent_buffer/get_batch: Retrieved batch w/ %ld samples, index_1=%d, z_ee,1^s=%f, dz_1=%f, g^1=%d, z_ee,1^s'=%f, r_1=%f",
                batch.size(),
                batch_indices.at(0),
                batch.at(0).state.endeffector_position.z,
                batch.at(0).action.relative_movement.at(2),
                batch.at(0).action.grasp,
                batch.at(0).next_state.endeffector_position.z,
                batch.at(0).reward.reward
            );
            return true;
        } else {
            ROS_INFO("agent_buffer/get_batch: Failed collecting samples for batch, empty batch returned");
            return false;
        }

    } else {
        
        ROS_INFO("agent_buffer/get_batch: Not enough samples collected(%d of min=%d)",
            replay_buffer.get_current_size(),
            MINIMUM_REPLAY_BUFFER_SIZE
        );
        return false;

    }

}

/**
 * @brief Initializes agent_buffer node by creating a server and a subscriber for add_sample topic and get_batch service. 
 * Additionally loads samples from file at specific location, if a file is found.
 * 
 * @param argc 
 * @param argv 
 * @return int status code
 */
int main(int argc, char **argv) {

    ros::init(argc, argv, "agent_buffer");
    ros::NodeHandle node;

    // Read all existing samples from file
    std::string filename(boost::archive::tmpdir());
    filename += FILE_PATH;
    filename += "/replay_buffer" + load_buffer_version + ".txt";
    try
    {
        // Create and open archive
        std::ifstream infilestream(filename.c_str());
        boost::archive::text_iarchive inarchive(infilestream);
        // Read class state from archive
        inarchive >> replay_buffer;
        for(int i=0; i<3; i++) {
            ROS_DEBUG("Read sample from file with gripper height: %f", replay_buffer.at(i).get_next_state_endeffector_position().z);
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    ROS_INFO("Restored %d samples from replay buffer file.", replay_buffer.get_current_size());

    // Create subscriber and server
    ros::Subscriber add_sample = node.subscribe("/agent_buffer/add_sample", 1000, add_sample_callback);
    ros::ServiceServer get_batch = node.advertiseService("/agent_buffer/get_batch", getBatch);

    ros::spin();
 
    return 0;

}
