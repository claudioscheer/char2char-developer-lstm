import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMModel, self).__init__()

        self.n_layers = 3
        self.hidden_size = 128

        self.lstm = nn.LSTM(
            input_size, self.hidden_size, self.n_layers, batch_first=True, dropout=0.5
        )
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        output, hidden_states = self.lstm(x)
        output = output.contiguous().view(-1, self.hidden_size)

        output = self.fc(output)

        return output, hidden_states
