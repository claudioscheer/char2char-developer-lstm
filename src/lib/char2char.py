import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Char2Char(nn.Module):
    def __init__(
        self,
        dictionary_size,
        embedding_size,
        lstm_hidden_size,
        number_lstm_layers,
        dropout_probability,
        output_size,
        device,
    ):
        super(Char2Char, self).__init__()

        self.device = device
        self.lstm_hidden_size = lstm_hidden_size
        self.number_lstm_layers = number_lstm_layers

        self.embedding = nn.Embedding(dictionary_size, embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=lstm_hidden_size,
            num_layers=number_lstm_layers,
            dropout=dropout_probability,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_probability)
        self.fc = nn.Linear(in_features=lstm_hidden_size, out_features=output_size)

    def forward(self, x, previous_hidden_states):
        x = self.embedding(x)
        x, hidden_states = self.lstm(x, previous_hidden_states)
        x = self.dropout(x)
        x = x.view(x.size()[0] * x.size()[1], self.lstm_hidden_size)
        x = self.fc(x)
        return x, hidden_states

    def predict(self, character, dataset, device, previous_hidden_states, top_k=5):
        x = np.array([[dataset.char2int[character]]])
        x = torch.from_numpy(x).to(device)

        previous_hidden_states = tuple([each.data for each in previous_hidden_states])
        output, hidden_states = self.forward(x, previous_hidden_states)

        p = F.softmax(output, dim=1).data.cpu()
        p, top_ch = p.topk(k=top_k)
        top_ch = top_ch.numpy().squeeze()
        p = p.numpy().squeeze()
        character = np.random.choice(top_ch, p=p / p.sum())

        return dataset.int2char[character], hidden_states

    def init_hidden(self, sequences_per_batch):
        return (
            torch.zeros(
                self.number_lstm_layers, sequences_per_batch, self.lstm_hidden_size
            ).to(self.device),
            torch.zeros(
                self.number_lstm_layers, sequences_per_batch, self.lstm_hidden_size
            ).to(self.device),
        )
