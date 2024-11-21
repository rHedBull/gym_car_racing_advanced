import os
from collections import deque

import cv2
import imageio
import numpy as np
import tensorflow as tf

import wandb
from matplotlib import pyplot as plt


def debug_display_image(image, step):
    """
    Helper function to display or save the image for debugging.
    """
    plt.imshow(image)  # Assuming `image` is in RGB format
    plt.title(f"Step: {step}")
    plt.axis('off')  # Remove axes for better visualization
    plt.show()  # Display the image inline

    # Optionally, save the image for offline debugging
    plt.savefig(f"debug_image_step_{step}.png")


class ExperimentLogger:
    def __init__(self, log_dir, experiment_name, window_size=100):
        """
        Initializes the ExperimentLogger.

        Args:
            log_dir (str): Base directory for logs.
            window_size (int): Number of episodes for rolling window metrics.
            experiment_name (str): Optional name for the experiment.
        """
        self.total_episodes = 0
        self.experiment_name = experiment_name

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = tf.summary.create_file_writer(log_dir)
        self.window_size = window_size

        # Hyperparameters storage
        self.hyperparams = {}

        # Metric tracking
        self.loss_history = deque(maxlen=100)
        self.q_value_history = deque(maxlen=100)
        self.gradient_norms = deque(maxlen=100)

        # Episode tracking
        self.episode_rewards = deque(maxlen=self.window_size)
        self.episode_steps = deque(maxlen=self.window_size)

        # Target network updates
        self.target_updates = 0

        # Image logging
        self.image_frames = []

    def log_step_metrics(self, loss, avg_q, gradient_norm):
        """
        Logs step-based metrics at defined intervals.

        Args:
            step (int): Current training step.
            loss (float): Current loss value.
            avg_q (float): Average Q-value.
            gradient_norm (float): Norm of gradients.
            buffer_size (int): Current size of the replay buffer.

        """
        self.loss_history.append(loss)
        self.q_value_history.append(avg_q)
        self.gradient_norms.append(gradient_norm)

        moving_avg_loss = np.mean(self.loss_history)
        moving_avg_q = np.mean(self.q_value_history)
        moving_avg_grad = np.mean(self.gradient_norms)

        wandb.log({
            "step/loss": loss,
            "step/average_loss": moving_avg_loss,
            "step/average_q": avg_q,
            "step/average_q_moving_avg": moving_avg_q,
            "step/gradient_norm": gradient_norm,
            "step/gradient_norm_moving_avg": moving_avg_grad
        }, step=step)

        # with self.writer.as_default():
        #     tf.summary.scalar("loss", loss, step)
        #     tf.summary.scalar("average_loss", moving_avg_loss, step)
        #     tf.summary.scalar("average_q", avg_q, step)
        #     tf.summary.scalar("average_q_moving_avg", moving_avg_q, step)
        #     tf.summary.scalar("gradient_norm", gradient_norm, step)
        #     tf.summary.scalar("gradient_norm_moving_avg", moving_avg_grad, step)
        #     tf.summary.scalar("buffer_size", buffer_size, step)

    def log_target_update(self):
        """
        Logs the occurrence of a target network update.

        Args:
            step (int): Current training step.
        """
        self.target_updates += 1

        wandb.log({
            "TargetNetwork/Updates": self.target_updates
        })
        # with self.writer.as_default():
        #     tf.summary.scalar("TargetNetwork/Updates", self.target_updates, step)

    def log_episode_metrics(self, step,  step_count, total_reward, epsilon, buffer_size):
        """
        Logs episode-based metrics.

        Args:
            :param step:
            :param buffer_size:
            :param total_reward:
            :param step_count:
            :param epsilon:
        """
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(step_count)
        average_reward = total_reward / step_count
        self.total_episodes += 1

        wandb.log({
            "episode/total_reward": total_reward,
            "episode/average_reward": average_reward,
            "episode/steps": step_count,
            "episode/epsilon": epsilon,
            "episode/buffer_size": buffer_size
        }, step=step)

    def log_evaluation_metrics(
        self,
        step_count,
        total_reward,
    ):
        average_reward = total_reward / step_count

        wandb.log({
            "eval/total_reward": total_reward,
            "eval/average_reward": average_reward
        })


        # with self.writer.as_default():
        #     tf.summary.scalar("Eval steps/episode", step_count, episode)
        #     tf.summary.scalar("Eval total reward/episode", total_reward, episode)
        #     tf.summary.scalar("Eval avrg. reward", average_reward, episode)

    def log_image(self, rgb_array, step):

        image_with_text = self._add_step_text(rgb_array, step)

        # transform observation to image
        tensor_img = tf.convert_to_tensor(image_with_text, dtype=tf.uint8)
        tensor_img = tf.expand_dims(tensor_img, 0)

        self.image_frames.append(image_with_text)

        with self.writer.as_default():
            tf.summary.image("Step: " + str(step), tensor_img, step)

    def create_gif(self, gif_name="evaluation.gif", max_frames=100):
        # Limit the number of frames to reduce memory usage
        n = max(1, len(self.image_frames) // max_frames)
        frames = self.image_frames[::n]

        # Create the GIF
        gif_path = os.path.join(self.log_dir, gif_name)
        imageio.mimsave(gif_path, frames, duration=0.1)  # duration specifies time per frame

        wandb.log({
            "Episode/GIF": wandb.Video(gif_path, format="gif")
        })
        os.remove(gif_path)

        # Clear the frames list to save memory
        self.image_frames = []

    def _add_step_text(self, image, step):
        """
        Adds the step number as text onto the image.
        """
        # Ensure the image is in the right format (uint8)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Convert to BGR format for OpenCV if image is RGB
        if image.shape[-1] == 3:  # Assuming last dimension is channels
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Add text (step number) to the image
        text = f"Step: {step}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # White color
        thickness = 2
        position = (10, 30)  # Top-left corner of the image
        cv2.putText(image, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

        # Convert back to RGB if needed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def close(self):
        """
        Closes the TensorBoard writer.
        """
        self.writer.close()
