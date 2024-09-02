import torch.nn as nn
import torch

# The Actor class defines a neural network model for strategy evaluation and contains the base part and three heads
class Actor(nn.Module):
    def __init__(self, num_states, num_points, num_channels, pmax):
        super(Actor, self).__init__()
        # Processing Incoming Status Information
        self.base = nn.Sequential(nn.Linear(num_states, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 128),
                                  nn.ReLU())
        # Output of the probability distribution of the target point
        self.point_header = nn.Sequential(nn.Linear(128, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, num_points),
                                          nn.Softmax(dim=-1))
        # Output channel probability distribution
        self.channel_header = nn.Sequential(nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, num_channels),
                                            nn.Softmax(dim=-1))
        # Average output power
        self.power_mu_header = nn.Sequential(nn.Linear(128, 64),
                                             nn.ReLU(),
                                             nn.Linear(64, 1),
                                             nn.Sigmoid())
        # Standard Deviation of Output Power
        self.power_sigma_header = nn.Sequential(nn.Linear(128, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 1),
                                                nn.Softplus())
        # Limit the range of power values generated
        self.pmax = pmax

    def forward(self, x):
        code = self.base(x)
        prob_points = self.point_header(code)
        prob_channels = self.channel_header(code)
        power_mu = self.power_mu_header(code) * (self.pmax - 1e-10) + 1e-10
        power_sigma = self.power_sigma_header(code)
        return prob_points, prob_channels, (power_mu, power_sigma)

# Define a critical model to evaluate the value function of the state(LSTM)
class Critic(nn.Module):
    def __init__(self, num_states, hidden_size=128, num_layers=1):  # Add the hidden_size and num_layers parameters.
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Add LSTM layer
        self.lstm = nn.LSTM(input_size=num_states, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # The output of the LSTM is mapped to state values through a linear layer
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64),
                                nn.ReLU(),
                                nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Initialize LSTM hidden and cellular states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step of the LSTM
        out = out[:, -1, :]
        # Map to state values using linear layers
        out = self.fc(out)
        return out