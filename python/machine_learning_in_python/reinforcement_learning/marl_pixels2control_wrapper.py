"""Wrapper placed around Gym Environments enabling easier multi-agent
reinforcement learning. Compatible with single-agent RL environments as well."""

import tensorflow as tf
import numpy as np

class ObservationWrapper:
    """ Class for stacking and processing frame observations.
    """

    def __init__(self, size=(96, 96), normalize=False, num_channels=3,
                 num_frames=1, num_agents=2):

        self.size = size  # Dimensions of observation frame
        self.normalize = normalize  # Normalize data from [0, 255] --> [0, 1]
        self.num_channels = num_channels  # 3 for RGB, 1 for greyscale
        self.num_frames = num_frames  # Number of frames in obs
        if self.num_frames > 1:  # Frame stacking
            self.frames = [tf.zeros(self.size + (self.num_channels,)) for i in
                                    range(self.num_frames)]  # Used as queue


    def get_obs_and_step(self, frame):
        """ Processes the observations from the environment.
        """

        processed_frame = self._process_image(tf.squeeze(frame))  # Process frame

        if self.num_frames == 1:  # Single-frame observations
            return processed_frame

        else:  # Frame stacking
            concat = [processed_frame] + self.frames[:-1]  # New frames list
            self.frames = concat  # Update frames
            stacked_frames = tf.concat(tuple(concat), axis=-1)  # Concatenate
            return stacked_frames

    def _process_image(self, image):
        """ Process each individual observation image.
        """
        if self.num_channels == 1:  # grayscale
            image = tf.image.rgb_to_grayscale(image)

        elif self.num_channels == 3:  # rgb
            if len(tf.shape(tf.squeeze(image)).numpy()) < 3:  # If grayscale
                image = tf.repeat(tf.expand_dims(image, axis=-1),
                                  self.num_channels, axis=-1)  # gray --> rgb

        input_size = tuple(tf.shape(image)[:2].numpy())  # Image (width, height)
        if input_size != self.size:
            kwargs = dict(
                output_shape=self.size, mode='edge', order=1,
                preserve_range=True)

            # Resize the image according to the size parameter
            image = tf.convert_to_tensor(resize(image, **kwargs).astype(np.float32))
            if self.normalize and np.max(image) > 1.0:  # [0, 255] --> [0, 1]
                image = tf.divide(image, 255.0)
        return image

    def reset(self):
        """ Method for resetting the observed frames. """
        if self.num_frames > 1:
            self.frames = [tf.zeros(self.size + (self.num_channels,)) for i in
                                    range(self.num_frames)]  # Used as queue
