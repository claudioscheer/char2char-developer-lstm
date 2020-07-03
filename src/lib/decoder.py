import torch
import torch.nn as nn
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(
        self,
        encoder_output,
        embedding_dim,
        lstm_hidden_size,
        number_lstm_layers,
        dropout_probability,
        output_size,
    ):
        super(Decoder, self).__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.number_lstm_layers = number_lstm_layers

        self.encoder = nn.Embedding(encoder_output, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_hidden_size,
            number_lstm_layers,
            dropout=dropout_probability,
        )
        self.dropout = nn.Dropout(dropout_probability)
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x, previous_hidden_states):
        x = self.encoder(x.view(1, -1))
        output, hidden_states = self.lstm(x.view(1, 1, -1), previous_hidden_states)
        output = self.dropout(output)
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        return output, hidden_states

    def init_hidden_states(self, batch_size, use_gpu=True):
        hidden_states = (
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
        )
        if use_gpu:
            return (hidden_states[0].cuda(), hidden_states[1].cuda())
        return hidden_states
