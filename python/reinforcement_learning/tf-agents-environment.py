"""A tf-agents environment wrapper for deep reinforcement learning.  
This enables for integration of tf-agents into this
custom framework for generating experience and training/testing with tf-agents.
"""

# Native Python imports
import os

# External package imports
import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import rospy  # to install on Ubuntu 18.04: sudo apt-get install -y python-rospy


# Include tf2-compatible functionality
tf.compat.v1.enable_v2_behavior()


class PySimulatorEnv(py_environment.PyEnvironment):
    """This class puts a tf-agents wrapper on the deep reinforcement learning 
    environment.  First, this environment is
    initialized as a Python environment, and it can later be converted into
    a TensorFlow environment using a tf-agents routine.
    Arguments:
        
        1. simulator (object): A simulator representation for your environment - 
                               e.g. an environment that carries a notion of 
                               state, and is capable of transitioning from 
                               one state to another.
                               
        2. discount_factor (float between [0,1]): This is the discount factor
            used for the Bellman equation, when calculating the reward.
            alpha (tuple): (Optional) A tuple of four float values
                corresponding to the penalty weights for each component of the
                loss.  The larger an alpha value, the larger its corresponding
                loss component will be.  RECOMMENDATION: Keep this 1.0 when
                training over long-length episodes.
        
        1. < OTHER ARGUMENTS > : Use other arguments that are specific to your 
                                 simulation environment here. 
    """

    def __init__(self, simualtor, discount_factor=1.0):
        self.timestep = None # Initialize the time step to be zero
        
        # These parameters need to be overwritten

        # 1. Action spec - i.e. what actions are allowed by the environment.
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(<INSERT_HERE>), dtype=np.float32, minimum=<INSERT_HERE>,
            maximum=<INSERT_HERE>, name='action')  # Actions specification

        # 2. Observation Spec - i.e. what is an ageng allowed to observe in this environment
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(<INSERT_HERE>, dtype=np.float32, minimum=<INSERT_HERE>,
            name='observation')  # States are [x y theta velocity]^T
        
        # 3. The "reset" state
        self.state0 = <INSERT_HERE>  # Store initial state for resets

        # 4. The general state
        self._state = <INSERT HERE> # Synchronize env <--> simulator

        # 5. Keep track of if an episode is completed
        self._episode_ended = False

        # 6. Discount factor
        self.discount_factor = discount_factor

    def action_spec(self):
        """Get action_spec class attributes.
        Getter method for the action_spec class attribute.
        Returns:
            Returns the action specification for this Python environment class.
        """

        return self._action_spec

    def observation_spec(self):
        """Get observation_spec class attributes.
        Getter method for the observation_spec class attribute.
        Returns:
            Returns the observation specification for this Python environment
            class.
        """

        return self._observation_spec

    def batch_size(self):
        return self.batch_size

    def batched(self):
        if self.batch_size() != 1:
            return True
        return False

    def _reset(self):
        """Reset the environment back to its default state.
        This method is used for resetting at the end of episodes,
        and returns the environment state to its initialized state.
        Returns:
            A tf-agents function that carries information about
            resetting relevant environment variables back to their default
            settings.
        """
        self._state = <PICK_RANDOM_STATE>  # Reset this to a random state
        self.timestep = 0  # Reset time step counter
        self._episode_ended = False
        return ts.restart(np.array(self.state0, dtype=np.float32))

    def _step(self, action):
        """Main functionality for stepping the RL model in this environment.
        This function lets the agent take an action, then updates the
        agent's state appropriately and computes the agent's reward.
        Arguments:
            action (list): A list corresponding to the action componets
                           [a1, a2, ... , aN] the agent takes at each time step.
        Returns:
            A tf-agents function that carries information about the
            current observation and discounted reward.
        """

        # If episode is over, after terminating time step, reset environment
        if self._episode_ended:
            return self.reset()

        # Else, step the agent, update state, and compute reward
        position_x, position_y, theta, velocity = \
            self.simulator.state_transition(self._state, action, self._dt)
        position_x, position_y = self.check_bounding_box([position_x,
                                                          position_y])
        # Update state here
        self._state = <UPDATE_STATE_HERE>

        # Compute reward
        reward = <INSERT REWARD COMPUTATION>

        # Check if the episode has ended
        if self.timestep.is_last():
            self._episode_ended = True
            return ts.termination(np.array(self._state,
                                           dtype=np.float32), reward=reward)

        # Else, step the time step counter and transition
        self.timestep += 1
        return ts.transition(np.array(self._state, dtype=np.float32),
                             reward=reward,
                             discount=float(self.discount_factor))