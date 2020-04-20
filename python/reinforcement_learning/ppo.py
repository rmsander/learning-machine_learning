# Native Python imports
import os
import copy
from datetime import datetime

# TensorFlow and tf-agents
from tf_agents.agents.ppo import ppo_agent, ppo_policy
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies.policy_saver import PolicySaver

# Other external packages
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import rospy  # to install on Ubuntu 18.04: sudo apt-get install -y python-rospy

# Get GPUs
from tensorflow.python.client import device_lib

def make_networks(env, conv_params=[(16, 8, 4), (32, 3, 2)]):
    """Function for creating the neural networks for the PPO agent, namely the actor and value networks.

    Source for network params: https://www.arconsis.com/unternehmen/blog/reinforcement-learning-doom-with-tf-agents-and-ppo

    Arguments:
        1. env (tf env): A TensorFlow environment that the agent interacts with via the neural networks.
        2. conv_params (list): A list corresponding to convolutional layer parameters for each neural network.

    Returns:
        1. actor_net (ActorDistributionNetwork): A tf-agents Actor Distribution Network that is used for action selection
                                                 with the PPO agent.
        2. value_net (ValueNetork): A tf-agents Value Network that is used for value estimation with the PPO agent.
    """
    # Define actor network
    actor_net = actor_distribution_network.ActorDistributionNetwork(env.observation_spec(), env.action_spec(),
                                                                    conv_layer_params=conv_params)
    # Define value network
    value_net = value_network.ValueNetwork(env.observation_spec(), conv_layer_params=conv_params)

    return actor_net, value_net


def make_agent(env, lr=8e-5):
    """Function for creating the TensorFlow PPO agent using the neural networks defined above.

    Arguments:
        1. env (tf env): A TensorFlow environment that the agent interacts with via the neural networks.
        2. lr (float): The learning rate for the PPO agent.

    Returns:
        1. agent (PPO agent): A PPO agent used for learning in the environment env.
    """
    # Create the actor-critic networks
    actor_net, value_net = make_networks(env)

    # Now create the PPO agent using the actor and value networks
    agent = ppo_agent.PPOAgent(
        env.time_step_spec(),
        env.action_spec(),
        actor_net=actor_net,
        value_net=value_net,
        optimizer=tf.compat.v1.train.AdamOptimizer(lr))
    return agent


class PPOTrainer:
    """
    A PPO trainer for tf-agents.  Uses PPO agent objects with TensorFlow environments to train agents to maxinimize
    reward in their environments.

    Arguments:
        1. ppo_agent (PPO agent): A PPO agent used for learning in the environment env.
        2. train_env (tf env): A TensorFlow environment that the agent interacts with via the neural networks.  Used for
                               creating training trajectories for the agent, and for optimizing its networks.
        3. eval_env (tf env): A TensorFlow environment that the agent interacts with via the neural networks.  Used for
                              evaluating the performance of the agent.

        5. use_tensorboard (bool): Whether or not to plot losses with tensorboard.
        4. add_training_to_video (bool): Whether or not to create videos of the agent's training and save them as videos.
    """
    def __init__(self, ppo_agent, train_env, eval_env, use_tensorboard=True, add_training_to_video=True):

        # Environment attributes
        self.train_env = train_env  # Environment for training
        self.eval_env = eval_env  # Environment for testing

        # Agent attributes
        self.agent = ppo_agent  # An instance of a tf-agents agent
        self.actor_net = self.agent._actor_net
        self.value_net = self.agent._value_net
        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy

        # Specifics of training
        self.max_buffer_size = 1000              # Collect entire memory buffer each time
        self.collect_steps_per_iteration = 1000  # Collect entire memory buffer each time
        self.epochs = 10000                       # Total number of episodes
        self.total_steps = self.epochs * self.collect_steps_per_iteration
        print("Total steps: {}".format(self.total_steps))

        # Evaluation
        self.num_eval_episodes = 5  # How many episodes we evaluate each time
        self.eval_returns = []      # Keep track of evaluation performance
        self.eval_interval = 100     # Evaluate every <x> epochs
        self.max_eval_episode_steps = 1000  # Most steps we can have in an episode

        # Logging
        self.time_ext = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_interval = 1
        self.policy_save_dir = os.path.join(os.getcwd(), "logging_{}/".format(self.time_ext))
        if not os.path.exists(self.policy_save_dir):
            print("Directory {} does not exist; creating it now".format(self.policy_save_dir))
            os.mkdir(self.policy_save_dir)
        self.video_train = []
        self.add_training_to_video = add_training_to_video
        self.video_eval = []

        # Tensorboard
        self.log_dir = "./tb_log_{}".format(self.time_ext)  # Log directory for tensorboard
        self.train_file_writer = tf.summary.create_file_writer(self.log_dir)  # File writer for tf
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.use_tensorboard = use_tensorboard  # Boolean for whether or not we use tensorboard for plotting


        # Create a replay buffer
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.max_buffer_size)

        # Get train and evaluation policy savers
        self.train_saver = PolicySaver(self.collect_policy, batch_size=None)
        self.eval_saver = PolicySaver(self.eval_policy, batch_size=None)

        # Specify directories for training and evaluation policies
        self.policy_save_dir = os.path.join(os.getcwd(), "models",
                                            self.time_ext)
        self.save_interval = 500  # Save every 100 epochs
        if not os.path.exists(self.policy_save_dir):
            print("Directory {} does not exist;"
                  " creating it now".format(self.policy_save_dir))
            os.makedirs(self.policy_save_dir, exist_ok=True)

    def make_checkpoints(self):
        """Function for creating checkpoints to save model and track progress."""
        global_step = tf.compat.v1.train.get_or_create_global_step()

        # Create a checkpoint for training
        self.train_checkpointer = common_utils.Checkpointer(
            ckpt_dir=self.policy_save_dir,
            agent=self.agent,
            global_step=global_step)

        # Create a readback checkpointer
        self.rb_checkpointer = common_utils.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
            max_to_keep=1,
            replay_buffer=self.replay_buffer)

    def collect_step(self, add_to_video=False, step=0, epoch=0):
        """
        Function for collecting a single step from the environment.  Used for adding trajectories to the replay
        buffer.  Resets on the first time step - indicating the start of a new episode.

        Arguments:
            1. add_to_video (bool): Whether or not to create a video of the training trajectories and save it to the
                                    'logging/' directory.
            2. step (int): The current step of the episode.  Important for determining whether or not the environment
                           needs to be reset and for tracking the training trajectories in tensorboard (if tensorboard
                           plotting is enabled).
            3. epoch (int): The current epoch of training.  Used for tracking the training trajectories in tensorboard
                            (if tensorboard plotting is enabled).
        """
        # Get current time step
        if step == 0:  # Reset the environment
            time_step = self.train_env.reset()
        else:  # Take the most recent time step
            time_step = self.train_env.current_time_step()

        # Take action using the collect policy
        action_step = self.collect_policy.action(time_step)

        # Compute the next time step by stepping the training environment
        next_time_step = self.train_env.step(action_step.action)

        # Create trajectory and write it to replay buffer
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        self.replay_buffer.add_batch(traj)

        # Log to tensorboard, if enabled
        if self.use_tensorboard:
            with self.train_file_writer.as_default():
                tf.summary.image("Training Trajectories, Epoch {}".format(epoch),
                                 time_step.observation, step=step)

        # Add observation to video, if enabled
        if add_to_video:
            # print(time_step.observation.numpy().shape)
            self.video_train.append(time_step.observation.numpy())

    def collect_episode(self, add_to_video=False, epoch=0):
        """
        Function for generating experience data for the replay buffer.  Calls collect_step() above to add trajectories
        from the environment to the replay buffer in an episodic fashion.  Trajectories from the replay buffer are then
        used for training the agent.

        Arguments:
            1. add_to_video (bool): Whether or not to create a video of the training trajectories and save it to the
                                    'logging/' directory.
            2. epoch (int): The current epoch of training.  Used for tracking the training trajectories in tensorboard
                            (if tensorboard plotting is enabled).
        """
        # Iteratively call collect_step method above to add trajectories to replay buffer
        for i in range(self.collect_steps_per_iteration):
            self.collect_step(add_to_video=add_to_video, step=i, epoch=epoch)

    def compute_avg_reward(self, epoch=None):
        """
        Function for computing the average reward over a series of evaluation episodes
        by creating simulation episodes using the agent's current policies,
        then computing rewards from taking actions using the evaluation (greedy) policy and averaging them.

        Arguments:
            1. epoch (int): The current epoch of training.  Used for tracking the training trajectories in tensorboard
                            (if tensorboard plotting is enabled).

        Returns:
            1. episode_return (float): A float representing the average reward over the interval of
            episodes which the agent's policies are evaluated.
        """
        total_return = 0.0
        for _ in range(self.num_eval_episodes):
            time_step = self.eval_env.reset()

            # Set step counter - capped at self.max_eval_episode_steps
            i = 0

            # Add to value in loop
            episode_return = 0.0

            while not time_step.is_last() and i < self.max_eval_episode_steps:
                action_step = self.eval_policy.action(time_step)
                self.video_eval.append(time_step.observation.numpy())  # Add to video frame
                time_step = self.eval_env.step(action_step.action)

                # Log to tensorboard
                if self.use_tensorboard:
                    with self.train_file_writer.as_default():
                        try:
                            tf.summary.image("Eval Trajectories, Epoch {}".format(epoch), time_step.observation, step=i)
                        except:
                            print("Please provide an input for the epoch number.")

                episode_return += time_step.reward
                if i % 250 == 0:
                    print("Action: {}, Reward: {}".format(action_step.action.numpy(), episode_return))
                i += 1
            print("Steps in episode: {}".format(i))
            total_return += episode_return
        avg_return = total_return / self.num_eval_episodes

        print("Average return: {}".format(avg_return))
        self.eval_returns.append(avg_return)
        return avg_return

    def train_agent(self):
        """
        Function for training a PPO tf-agent using trajectories from the replay buffer.  Does initial evaluation of the
        agent prior to training, and then iterates over epochs of the following procedure:

            a. Collect an episode of data, and write the trajectories to the replay buffer.
            b. Train from the trajectories on the replay buffer.  Updates the weights of the actor and value networks.
            c. Empty the replay buffer.
            d. (If enabled) Save data to disk for tensorboard.
            e. Depending on epoch number and the evaluation and logging intervals, evaluate the agent or log information.

        Returns:
            1. agent (PPO agent): The PPO agent trained during the training process
        """
        eval_epochs = []

        # Optimize by wrapping some of the code in a graph using TF function.
        self.agent.train = common.function(self.agent.train)
        self.agent.train_step_counter.assign(0)
        avg_return = self.compute_avg_reward(epoch=0)  # Compute pre-training metrics

        # Log average reward to tensorboard
        if self.use_tensorboard:
            with self.train_file_writer.as_default():
                tf.summary.scalar("Avg. Reward", float(avg_return), step=0)

        print("DONE WITH PRELIMINARY EVALUATION...")
        # Append for output plot
        eval_epochs.append(0)
        self.video_eval = []  # Empty to create a new eval video
        returns = [avg_return]

        time_step = self.train_env.reset()

        # Episode counter
        i = 0
        for i in range(self.epochs):
            print("Training epoch: {}".format(i))

            # Collect data and train agent; clear buffer at end
            print("COLLECTING EPISODE")
            # Reset the old training video
            self.video_train = []
            self.collect_episode(add_to_video=self.add_training_to_video, epoch=i)
            self.create_video(mode='train', ext=i)
            print("COLLECTED EPISODE")
            trajectories = self.replay_buffer.gather_all()

            # Old weights
            old_vnet = copy.deepcopy(
                self.agent._value_net.trainable_variables[0])
            old_anet = copy.deepcopy(
                self.agent._actor_net.trainable_variables[0])

            # Take training step
            train_loss = self.agent.train(experience=trajectories)

            # Log loss to tensorboard
            if self.use_tensorboard:
                with self.train_file_writer.as_default():
                    tf.summary.scalar("Training Loss", float(train_loss.loss), step=i)

            # Get new weights
            new_vnet = copy.deepcopy(
                self.agent._value_net.trainable_variables[0])
            new_anet = copy.deepcopy(
                self.agent._actor_net.trainable_variables[0])

            # Display Frobenius norm
            print("VALUE NET Frobenius Norm Difference: {}".format(tf.norm(
                old_vnet - new_vnet)))
            print("ACTOR NET Frobenius Norm Difference: {}".format(tf.norm(
                old_anet - new_anet)))

            # Step the counter, and log/evaluate agent
            step = self.agent.train_step_counter.numpy()

            if self.epochs % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss.loss))

            if (i+1) % self.eval_interval == 0:

                avg_return = self.compute_avg_reward(epoch=i)

                # Log average reward to tensorboard
                if self.use_tensorboard:
                    with self.train_file_writer.as_default():
                        tf.summary.scalar("Avg. Reward", float(avg_return), step=i)

                eval_epochs.append(i + 1)
                print(
                    'epoch = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)
                self.create_video(mode='eval', ext=i)
                self.video_eval = []  # Empty to create a new eval video

            # We should save checkpoints every save_interval epochs
            if i % self.save_interval == 0 and i != 0:
                self.save_policy(epochs_done=i)
                print("Epochs: {}".format(i))

            self.replay_buffer.clear()

        # At the end of training, return the agent
        return self.agent

    def playback_trajectories(self, recdir=None):

        counts = []

        def handle_ep(observations, actions, rewards):
            counts[0] += 1
            counts[1] += observations.shape[0]
            logger.debug('Observations.shape={}, actions.shape={}, rewards.shape={}', observations.shape, actions.shape,
                         rewards.shape)

        if recdir is None:
            print("Error: Please specify a recording directory by calling gym_env.directory")
        else:
            scan_recorded_traces(recdir, handle_ep)

    def create_video(self, mode='eval', ext=0):
        if mode == 'eval':
            video = self.video_eval
        elif mode == 'train':
            video = self.video_train
        # Check if video is zero length
        if len(video) == 0:
            raise AssertionError("Video is empty.")
        print("Number of frames in video: {}".format(len(video)))
        obs_size = video[0].shape
        width = np.uint(obs_size[-3])
        height = np.uint(obs_size[-2])
        channels = np.uint(obs_size[-1])
        print("HEIGHT IS: {}, WIDTH IS: {}, CHANNELS IS: {}".format(width, height, channels))
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(
            os.path.join(self.policy_save_dir, "trajectories_{}_epoch_{}.avi".format(mode, ext)), fourcc, self.FPS, (width, height))
        for i in range(len(video)):
            img_rgb = cv.cvtColor(np.uint8(255 * video[i][0]), cv.COLOR_BGR2RGB)  # Save as RGB image
            out.write(img_rgb)
        out.release()

    def plot_eval(self):
        xs = [i * self.eval_interval for i in range(len(self.eval_returns))]
        plt.plot(xs, self.eval_returns)
        plt.xlabel("Training epochs")
        plt.ylabel("Average Return")
        plt.title("Average Returns as a Function of Training")
        plt.savefig(os.path.join(self.policy_save_dir, "eval_returns.png"))
        print("CREATED PLOT OF RETURNS...")

    def save_policy(self, epochs_done=0):
        """
        Using the PolicySaver(s) defined in the trainer constructor, this
        function saves the training and evaluation policies according to the
        policy_save_dir attribute and whether multiple PPO agents or a single
        master PPO agent is used.

        Arguments:
            1. epochs_done (int):  The number of epochs completed in the
                                   training process at the time this save
                                   function is called.
        """

        # Save training policy
        train_save_dir = os.path.join(self.policy_save_dir, "train",
                                      "epochs_{}".format(epochs_done))
        if not os.path.exists(train_save_dir):
            os.makedirs(train_save_dir, exist_ok=True)
        self.train_saver.save(train_save_dir)

        print("Training policy saved...")

        # Save eval policy
        eval_save_dir = os.path.join(self.policy_save_dir, "eval",
                                     "epochs_{}".format(epochs_done))
        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir, exist_ok=True)
        self.eval_saver.save(eval_save_dir)

        print("Eval policy saved...")

    def load_saved_policy(self, eval_model_path=None, train_model_path=None):

        # Load evaluation and/or training policies from path
        if eval_model_path is not None:
            self.eval_policy = tf.saved_model.load(eval_model_path)
            print("Loading evaluation policy from: {}".format(eval_model_path))

        if train_model_path is not None:
            self.collect_policy = tf.saved_model.load(train_model_path)
            print("Loading training policy from: {}".format(train_model_path))

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def main():
    """
    Main function for creating a PPO agent and training it on the designated
    environment.
    """
    ros_env = None # ROS ENVIRONMENT GOES HERE

    # Conv2d layers don't like transforming tf.float32 into tf.uint8
    ros_env.observation_space.dtype = np.float32

    # Now create Python and TensorFlow environments from ros environment
    train_env = tf_py_environment.TFPyEnvironment(ros_env)  # Python --> TensorFlow

    # Display environment specs
    print("Observation spec: {} \n".format(train_env.observation_spec()))
    print("Action spec: {} \n".format(train_env.action_spec()))
    print("Time step spec: {} \n".format(train_env.time_step_spec()))
    print("Training env spec: {}".format(train_env.observation_spec()))

    # Create an evaluation environment for evaluating PPO agent
    eval_env = tf_py_environment.TFPyEnvironment(ros_env)

    # Instantiate and initialize the agent
    agent = make_agent(train_env)
    agent.initialize()

    # Instantiate the trainer
    trainer = PPOTrainer(agent, train_env, eval_env, downsample=DOWNSAMPLE)

    print("Initialized agent, beginning training...")

    # Train agent, and when finished, save model
    trained_agent = trainer.train_agent()

    print("Training finished; video playback...")
    trainer.plot_eval()

    print("Training finished; saving agent...")

    trainer.save_policy()
    print("Agent saved.")




if __name__ == "__main__":
    main()
