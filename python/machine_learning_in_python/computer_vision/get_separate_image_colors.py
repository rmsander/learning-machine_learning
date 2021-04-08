import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from copy import deepcopy

# Load image and convert it from BGR (opencv default) to RGB
fpath = "dog.png"  # TODO: replace with your path
IMG = cv.cvtColor(cv.imread(fpath), cv.COLOR_BGR2RGB)

# Get dimensions and reshape into (H * W, C) vector - i.e. a long vector, where each element is a tuple corresponding to a color!
H, W, C = IMG.shape
IMG_FLATTENED = np.vstack([IMG[:, w, :] for w in range(W)])

# Get unique colors using np.unique function, and their counts
colors, counts = np.unique(IMG_FLATTENED, axis=0, return_counts = True)

# Jointly loop through colors and counts
for color, count in zip(colors, counts):

    print("COLOR: {}, COUNT: {}".format(color, count))
    # Create placeholder image and mark all pixels as white
    SINGLE_COLOR = (255 * np.ones(IMG.shape)).astype(np.uint8)  # Make sure casted to uint8

    # Compute binary mask of pixel locations where color is, and set color in new image
    color_idx = np.all(IMG[..., :] == color, axis=-1)
    SINGLE_COLOR[color_idx, :] = color

    # Write file to output with color and counts specified
    cv.imwrite("color={}_count={}.png".format(color, count), SINGLE_COLOR)
