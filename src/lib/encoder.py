import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(
        self,
        dictionary_size,
        embedding_dim,
        lstm_hidden_size,
        number_lstm_layers,
        dropout_probability,
    ):
        super(Encoder, self).__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.number_lstm_layers = number_lstm_layers

        self.embedding = nn.Embedding(dictionary_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_hidden_size,
            number_lstm_layers,
            dropout=dropout_probability,
        )
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        output = self.embedding(x)
        output, hidden_states = self.lstm(output)
        output = self.dropout(output)
        return hidden_states
