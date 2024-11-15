import os
from collections import deque

import numpy as np
import tensorflow as tf


class ExperimentLogger:
    def __init__(self, log_dir, experiment_name, window_size=100):
        """
        Initializes the ExperimentLogger.

        Args:
            log_dir (str): Base directory for logs.
            window_size (int): Number of episodes for rolling window metrics.
            experiment_name (str): Optional name for the experiment.
        """
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

    def log_hyperparameters(self, hyperparams):
        """
        Logs the hyperparameters of the experiment.

        Args:
            hyperparams (dict): Dictionary of hyperparameters.
        """
        self.hyperparams = hyperparams
        for key, value in hyperparams.items():
            self.writer.add_text("Hyperparameters", f"{key}: {value}\n")
        # Optionally, save hyperparameters to a file
        with open(os.path.join(self.log_dir, "hyperparameters.txt"), "w") as f:
            for key, value in hyperparams.items():
                f.write(f"{key}: {value}\n")

    def log_step_metrics(self, step, loss, avg_q, gradient_norm, buffer_size):
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

        with self.writer.as_default():
            tf.summary.scalar("loss", loss, step)
            tf.summary.scalar("average_loss", moving_avg_loss, step)
            tf.summary.scalar("average_q", avg_q, step)
            tf.summary.scalar("average_q_moving_avg", moving_avg_q, step)
            tf.summary.scalar("gradient_norm", gradient_norm, step)
            tf.summary.scalar("gradient_norm_moving_avg", moving_avg_grad, step)
            tf.summary.scalar("buffer_size", buffer_size, step)

    def log_target_update(self, step):
        """
        Logs the occurrence of a target network update.

        Args:
            step (int): Current training step.
        """
        self.target_updates += 1
        with self.writer.as_default():
            tf.summary.scalar("TargetNetwork/Updates", self.target_updates, step)

    def log_episode_metrics(self, episode, step_count, total_reward):
        """
        Logs episode-based metrics.

        Args:
            episode (int): Current episode number.
            total_reward (float): Total reward accumulated in the episode.
            step_count (int): Number of steps taken in the episode.
            success (bool): Whether the episode was successful.
        """
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(step_count)

        # Log per-episode metrics
        with self.writer.as_default():
            tf.summary.scalar("Reward/Episode", total_reward, episode)
            tf.summary.scalar("Steps/Episode", step_count, episode)

        # Log rolling window metrics
        if len(self.episode_rewards) == self.window_size:
            avg_reward = np.mean(self.episode_rewards)
            avg_steps = np.mean(self.episode_steps)

            with self.writer.as_default():
                tf.summary.scalar("Reward/Average_Window", avg_reward, episode)
                tf.summary.scalar("Steps/Average_Window", avg_steps, episode)

    def log_evaluation_metrics(
        self,
        episode,
        step_count,
        total_reward,
    ):
        average_reward = total_reward / step_count
        with self.writer.as_default():
            tf.summary.scalar("Eval steps/episode", step_count, episode)
            tf.summary.scalar("Eval total reward/episode", total_reward, episode)
            tf.summary.scalar("Eval avrg. reward", average_reward, episode)

    def close(self):
        """
        Closes the TensorBoard writer.
        """
        self.writer.close()
