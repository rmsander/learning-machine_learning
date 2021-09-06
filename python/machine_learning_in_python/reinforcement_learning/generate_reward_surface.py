"""Functions and script to generate a reward surface with state and
action inputs. Configured for the Pendulum-v0 Gym environment."""

import gym
import matplotlib.pyplot as plt
import numpy as np

# Parameters
SAMPLES = 50
SEED = 42
BOUNDS = [(-np.pi, np.pi), (-8, 8), (-2, 2)]

# Set seeds
np.random.seed(seed=SEED)  # Set seed for NumPy RNG

def sample_uniformly(N, bounds):
  """Utility function for sampling points iid from 3D space."""

  # Get bounds
  [(low_s1, high_s1), (low_s2, high_s2), (low_a, high_a)] = bounds
  N_cubed = N**3

  # Now sample each dimension
  s1 = np.random.uniform(low=low_s1, high=high_s1, size=N_cubed)
  s2 = np.random.uniform(low=low_s2, high=high_s2, size=N_cubed)
  a = np.random.uniform(low=low_a, high=high_a, size=N_cubed)

  # Now combine
  X = np.stack((s1, s2, a), axis=-1)
  Y = np.zeros(N_cubed)
  for k in range(N_cubed):  # Iterate over generated samples
    Y[k] = compute_reward(X[k])

  return X, Y

def plot_points(X, Y, title="Analytic Reward Over State-Action Space"):
  """Utility function for plotting points."""
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel('Theta (th)')
  ax.set_ylabel('Theta dot (thdot)')
  ax.set_zlabel('Action (u)')

  img = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.hot())
  fig.colorbar(img, orientation="horizontal", pad=0.03)
  fig.tight_layout()
  plt.title(title)
  plt.show()

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def compute_reward(x):
  th, thdot, u = x
  """Analytic function to compute reward given states and action."""
  return -(angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2))

def generate_lin_space_samples(N, bounds):
  """Function for generating samples from a linear space."""

  # Get bounds
  [(low_s1, high_s1), (low_s2, high_s2), (low_a, high_a)] = bounds

  # Step sizes
  step_s1 = (high_s1 - low_s1) / N
  step_s2 = (high_s2 - low_s2) / N
  step_a = (high_a - low_a) / N

  # Create space
  X = np.mgrid[low_s1:high_s1:step_s1,
                    low_s2:high_s2:step_s2,
                    low_a:high_a:step_a].reshape(3, -1).T

  # Generate rewards
  N_cubed = N**3
  Y = np.zeros(N_cubed)
  for k in range(N_cubed):  # Iterate over generated samples
      Y[k] = compute_reward(X[k])


  # Now compute the reward at each of these points
  return X, Y

def main():
  X, Y = generate_lin_space_samples(SAMPLES, BOUNDS)
  plot_points(X, Y)

if __name__ == "__main__":
  main()
