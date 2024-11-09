from QNetwork import QNetwork
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQN:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10
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

        # Q-Network and Target Network
        self.q_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is not trained

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.steps_done = 0
        self.target_update_freq = target_update_freq

    def select_action(self, state):
        """Selects an action using ε-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Stores a transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        """Samples a batch of transitions from memory."""
        return random.sample(self.memory, self.batch_size)

    def train_step(self):
        """Performs a single training step."""
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to train

        batch = self.sample_memory()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)  # Shape: [batch, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions)

        # Target Q-values
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path):
        """Saves the Q-network's state."""
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        """Loads the Q-network's state."""
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
