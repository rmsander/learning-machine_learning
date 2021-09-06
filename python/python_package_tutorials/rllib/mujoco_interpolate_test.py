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
        obs = env.step(a)[0]
        states.append(env.state_vector())
        Qpos.append(env.sim.get_state().qpos)
        Qvel.append(env.sim.get_state().qvel)

    # Now, we need to set the state of the simulator
    for i in range(10):

        # Set the state according to original state
        env.set_state(Qpos[i], Qvel[i])

        # Set the state, and then check if it's equal to the original state
        current_state = env.state_vector()

        assert np.equal(states[i], current_state).all()
        print("Current state of env (i = {}): "
              "\n{} \nStored State of env: (i = {}) \n{}".format(
            i, current_state, i, states[i]
        ))

    # Now combine states 1 and 2, and qpos/qvel 1/2
    qpos_12 = 0.5 * Qpos[0] + 0.5 * Qpos[1]
    qvel_12 = 0.5 * Qvel[0] + 0.5 * Qvel[1]

    # State combine
    state_12 = 0.5 * states[0] + 0.5 * states[1]
    print("________")
    print("STATE 12: \n{}".format(state_12))

    # Now combine again
    env.set_state(qpos_12, qvel_12)
    print("TEST 12: \n{}".format(env.state_vector()))
    assert env.state_vector() == state_12

if __name__ == '__main__':
    main()
