import math
import torch

import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)
        return context_vector, attention_weights


class ImprovedGaitLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(ImprovedGaitLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 2)  # Output cosine and sine

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            x.device
        )  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Get the LSTM hidden states for all timesteps
        out, _ = self.lstm(x, (h0, c0))  # [B, 12, hidden_size * 2]

        # Pass each timestep's hidden state through fc1 and fc2
        out = self.fc1(out)  # [B, 12, hidden_size]
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # [B, 12, 2]

        # Normalize the output for each timestep to ensure it's on the unit circle
        norm = torch.norm(out, dim=2, keepdim=True)
        out = out / norm

        return out  # [B, 12, 2]

    def get_phase(self, x):
        cos_sin = self.forward(x)  # Output shape: [B, 12, 2]
        phase = torch.atan2(cos_sin[:, :, 1], cos_sin[:, :, 0]) / (2 * math.pi)  # [B, 12]
        phase = torch.where(
            phase < 0, phase + 1, phase
        )  # Ensure phase is between 0 and 1
        return phase  # [B, 12]
