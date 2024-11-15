from QNetwork import QNetwork
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np


def device():
    """Returns the device to run computations on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN:
    def __init__(
        self,
        logger,
        state_size,
        action_size,
        hidden_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.999934,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=1000,
        checkpoint_dir='checkpoint',
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # Epsilon parameters for ε-greedy policy
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Replay memory
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Network and Target Network
        self.q_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is not trained

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss(reduction='mean')

        self.steps_done = 0
        self.target_update_freq = target_update_freq

        self.logger = logger

        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)  # Create directory if it doesn't exist

    def select_action(self, state):
        """Selects an action using ε-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def save_checkpoint(self, filename=None):
        """Saves the model and optimizer states."""
        if filename is None:
            filename = f"checkpoint_{self.steps_done}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, filepath):
        """Loads the model and optimizer states."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No checkpoint found at '{filepath}'")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded: {filepath}")

    def store_transition(self, state, action, reward, next_state, done):
        """Stores a transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        """Samples a batch of transitions from memory."""
        return random.sample(self.memory, self.batch_size)

    def train_step(self):
        """Performs a single training step."""
        if len(self.memory) < self.batch_size:
            return 0 # Not enough samples to train

        batch = self.sample_memory()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).unsqueeze(1).to(device())  # Shape: [batch, 1, 96, 96]
        actions = torch.LongTensor(actions).unsqueeze(1).to(device())  # Shape: [batch, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device())  # Shape: [batch, 1]
        next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(1).to(device())  # Shape: [batch, 1, 96, 96]
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device())

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions)

        # Target Q-values
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)  # Shape: [batch, 1]
            max_next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute gradient norms
        total_norm = 0
        for param in self.q_network.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        avg_q = current_q.mean().item()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.logger.log_target_update(self.steps_done)

        self.logger.log_step_metrics(self.steps_done, loss.item(), avg_q, total_norm, len(self.memory))

    def save_model(self, path):
        """Saves the Q-network's state."""
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        """Loads the Q-network's state."""
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
