import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMModel, self).__init__()

        self.n_layers = 2
        self.hidden_size = 64

        self.lstm = nn.LSTM(
            input_size, self.hidden_size, self.n_layers, batch_first=True, dropout=0.5
        )
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, previous_hidden_states):
        output, hidden_states = self.lstm(x, previous_hidden_states)
        output = output.contiguous().view(-1, self.hidden_size)

        output = self.fc(output)

        return output, hidden_states

    def init_hidden_states(self, batch_size, use_gpu=True):
        hidden_states = (
            torch.zeros(self.n_layers, batch_size, self.hidden_size),
            torch.zeros(self.n_layers, batch_size, self.hidden_size),
        )
        if use_gpu:
            return (hidden_states[0].cuda(), hidden_states[1].cuda())
        return hidden_states
