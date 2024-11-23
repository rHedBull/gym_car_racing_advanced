import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=8, stride=4
        )  # Output: 32 x 23 x 23
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2
        )  # Output: 64 x 10 x 10
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1
        )  # Output: 64 x 8 x 8

        # Calculate the size of the input to the first fully connected layer
        # After Conv3: 64 channels, 8x8 feature map
        self.fc_input_dim = 64 * 8 * 8  # 4096

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, hidden_size)
        self.fc2 = nn.Linear(
            hidden_size, action_size
        )  # Output: Q-value for each action

    def forward(self, state):
        """
        Forward pass through the network.

        Parameters:
        - state: Tensor of shape (batch_size, 1, 96, 96)

        Returns:
        - Q-values: Tensor of shape (batch_size, action_size)
        """
        x = F.relu(self.conv1(state))  # Shape: (batch_size, 32, 23, 23)
        x = F.relu(self.conv2(x))  # Shape: (batch_size, 64, 10, 10)
        x = F.relu(self.conv3(x))  # Shape: (batch_size, 64, 8, 8)
        x = x.view(-1, self.fc_input_dim)  # Flatten: (batch_size, 4096)
        x = F.relu(self.fc1(x))  # Shape: (batch_size, 512)
        q_values = self.fc2(x)  # Shape: (batch_size, 20)
        return q_values
