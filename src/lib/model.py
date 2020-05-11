import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMModel, self).__init__()

        self.n_layers = 3
        self.hidden_size = 256

        self.lstm = nn.LSTM(
            input_size, self.hidden_size, self.n_layers, batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        # batch_size = x.size(axis=0)
        # initial_hidden_state = self.init_hidden_state(batch_size)
        # print(initial_hidden_state.shape)
        # print(x.shape)

        output, hidden_states = self.lstm(x)
        output = output.contiguous().view(-1, self.hidden_size)

        output = self.fc(output)

        return output, hidden_states

    def init_hidden_state(self, batch_size):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden_state
