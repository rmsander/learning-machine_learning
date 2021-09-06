"""Setup environments from Deepmind Control (DMC) for training agents
in the RLlib Python RL library.
"""

# Global variables
FRAME_SKIP = 1  # Skip these frames consecutively
FROM_PIXELS = False  # False --> state-based, True --> pixels-to-control


from ray.rllib.env.dm_control_wrapper import DMCEnv
from parameters import FROM_PIXELS, FRAME_SKIP


def acrobot_swingup(from_pixels=FROM_PIXELS,
                    height=64,
                    width=64,
                    frame_skip=FRAME_SKIP,
                    channels_first=True):
    return DMCEnv(
        "acrobot",
        "swingup",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def walker_walk(from_pixels=FROM_PIXELS,
                height=64,
                width=64,
                frame_skip=FRAME_SKIP,
                channels_first=True):
    return DMCEnv(
        "walker",
        "walk",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def hopper_hop(from_pixels=FROM_PIXELS,
               height=64,
               width=64,
               frame_skip=FRAME_SKIP,
               channels_first=True):
    return DMCEnv(
        "hopper",
        "hop",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def hopper_stand(from_pixels=FROM_PIXELS,
                 height=64,
                 width=64,
                 frame_skip=FRAME_SKIP,
                 channels_first=True):
    return DMCEnv(
        "hopper",
        "stand",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def cheetah_run(from_pixels=FROM_PIXELS,
                height=64,
                width=64,
                frame_skip=FRAME_SKIP,
                channels_first=True):
    return DMCEnv(
        "cheetah",
        "run",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def walker_run(from_pixels=FROM_PIXELS,
               height=64,
               width=64,
               frame_skip=FRAME_SKIP,
               channels_first=True):
    return DMCEnv(
        "walker",
        "run",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def pendulum_swingup(from_pixels=FROM_PIXELS,
                     height=64,
                     width=64,
                     frame_skip=FRAME_SKIP,
                     channels_first=True):
    return DMCEnv(
        "pendulum",
        "swingup",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def cartpole_swingup(from_pixels=FROM_PIXELS,
                     height=64,
                     width=64,
                     frame_skip=FRAME_SKIP,
                     channels_first=True):
    return DMCEnv(
        "cartpole",
        "swingup",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)


def humanoid_walk(from_pixels=FROM_PIXELS,
                  height=64,
                  width=64,
                  frame_skip=FRAME_SKIP,
                  channels_first=True):
    return DMCEnv(
        "humanoid",
        "walk",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)

def quadruped_run(from_pixels=FROM_PIXELS,
                  height=64,
                  width=64,
                  frame_skip=FRAME_SKIP,
                  channels_first=True):
    return DMCEnv(
        "quadruped",
        "run",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)

def quadruped_walk(from_pixels=FROM_PIXELS,
                  height=64,
                  width=64,
                  frame_skip=FRAME_SKIP,
                  channels_first=True):
    return DMCEnv(
        "quadruped",
        "walk",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)

def finger_spin(from_pixels=FROM_PIXELS,
                  height=64,
                  width=64,
                  frame_skip=FRAME_SKIP,
                  channels_first=True):
    return DMCEnv(
        "finger",
        "spin",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)

def cup_catch(from_pixels=FROM_PIXELS,
                  height=64,
                  width=64,
                  frame_skip=FRAME_SKIP,
                  channels_first=True):
    return DMCEnv(
        "ball_in_cup",
        "catch",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)

def reacher_easy(from_pixels=FROM_PIXELS,
                  height=64,
                  width=64,
                  frame_skip=FRAME_SKIP,
                  channels_first=True):
    return DMCEnv(
        "reacher",
        "easy",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)

def reacher_hard(from_pixels=FROM_PIXELS,
                  height=64,
                  width=64,
                  frame_skip=FRAME_SKIP,
                  channels_first=True):
    return DMCEnv(
        "reacher",
        "hard",
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        channels_first=channels_first)
