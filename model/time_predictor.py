import torch.nn as nn


class StrideTimeEstimatorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StrideTimeEstimatorLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last hidden state
        return out
