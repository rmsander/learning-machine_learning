"""Module for testing whether interpolation of qpos and qvel corresponds to
interpolation of states/observations in MuJoCo environments.
Conclusion: it does!"""

from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import numpy as np


def main():
    # Initialize and reset the environment to extract initial state
    env = HalfCheetahEnv()
    state = env.reset()

    # Get qpos and qvel
    Qpos = [env.sim.get_state().qpos]
    Qvel = [env.sim.get_state().qvel]

    # Get state
    states = [env.state_vector()]

    # Step the model and record data
    for i in range(10):
        a = env.action_space.sample()
        env.step(a)
        states.append(env.state_vector())
        Qpos.append(env.sim.get_state().qpos)
        Qvel.append(env.sim.get_state().qvel)

    # Now, we need to set the state of the simulator
    for i in range(10):

        # Set the state according to original state
        env.set_state(Qpos[i], Qvel[i])

        # Set the state, and then check if it's equal to the original state
        current_state = env.state_vector()

        # Checks that current state of env is equal to stored state
        assert np.equal(states[i], current_state).all()
        print("Current state of env (i = {}): "
              "\n{} \nStored State of env: (i = {}) \n{}".format(
            i, current_state, i, states[i]
        ))

if __name__ == '__main__':
    main()
