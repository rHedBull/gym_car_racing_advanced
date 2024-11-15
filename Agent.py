import random
from itertools import product

import cv2
from matplotlib import pyplot as plt

from DQN import DQN

# continuous there are 3 actions :
#
# 0: steering, -1 is full left, +1 is full right
#
# 1: gas
#
# 2: breaking

steering_options = [-1.0, -0.5, 0.0, 0.5, 1.0]
gas_options = [0, 1]
break_options = [0, 1]

state_size = 96 * 96


class Agent:
    def __init__(self, hyperparameters, logger, model_path=None):
        self.reward = None
        self.state = None

        # Generate all possible combinations
        action_combinations = list(
            product(steering_options, gas_options, break_options)
        )

        # Assign a unique index to each combination
        self.action_mapping = {
            idx: list(action) for idx, action in enumerate(action_combinations)
        }

        action_size = len(self.action_mapping)

        self.DQN = DQN(hyperparameters, state_size, action_size, logger)
        if model_path is not None:
            self.DQN.load_model(model_path)

        self.reset()

    def reset(self):
        self.reward = 0
        self.state = "active"

    def get_action(self, observation):
        """Selects an action using the DQN's policy."""
        grey_obs = rgb_to_grayscale_opencv(observation)
        action_index = self.DQN.select_action(grey_obs)
        action_values = self.action_mapping.get(action_index)
        return action_values

    def store_transition(self, old_observation, action, reward, new_observation, done):
        """Stores a transition in the DQN's replay buffer."""

        # convert both observations to grayscale
        old_observation = rgb_to_grayscale_opencv(old_observation)
        new_observation = rgb_to_grayscale_opencv(new_observation)
        action_index = self.get_action_index(action)
        self.DQN.store_transition(
            old_observation, action_index, reward, new_observation, done
        )

    def update(self, reward):
        self.reward += reward

    def train(self):
        return self.DQN.train_step()

    def save_model(self, path):
        self.DQN.save_model(path)

    def load_model(self, path):
        self.DQN.load_model(path)

    def get_action_index(self, action):
        for idx, act in self.action_mapping.items():
            if act == action:
                return idx
        return None

    def save_checkpoint(self, current_episode, total_episodes, checkpoint_path=None):
        self.DQN.save_checkpoint(current_episode, total_episodes, checkpoint_path)


def get_random_action():
    return [random.uniform(-1, 1), random.uniform(0, 1), random.uniform(0, 1)]


def rgb_to_grayscale_opencv(rgb):
    """
    Convert an RGB image to grayscale using OpenCV.

    Parameters:
        rgb (numpy.ndarray): Input image array with shape (height, width, 3).

    Returns:
        numpy.ndarray: Grayscale image array with shape (height, width).
    """
    # OpenCV expects the image in BGR format by default
    # If your image is RGB, convert it to BGR first
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    grayscale = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return grayscale


def print_obs(img, grey_obs):
    # Validate the image shape
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected observation[0] to be an RGB image with 3 channels.")

    # Display the original RGB image
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original RGB Image")
    plt.axis("off")  # Hide axis

    # Display the grayscale image
    plt.subplot(1, 2, 2)
    plt.imshow(grey_obs, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")  # Hide axis

    plt.tight_layout()
    plt.show()
