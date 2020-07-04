import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Char2Char(nn.Module):
    def __init__(
        self,
        dictionary_size,
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
        self.dictionary_size = dictionary_size

        self.lstm = nn.LSTM(
            input_size=dictionary_size,
            hidden_size=lstm_hidden_size,
            num_layers=number_lstm_layers,
            dropout=dropout_probability,
        )
        self.dropout = nn.Dropout(dropout_probability)
        self.fc = nn.Linear(in_features=lstm_hidden_size, out_features=output_size)

    def forward(self, x, previous_hidden_states):
        x, (h, c) = self.lstm(x, previous_hidden_states)
        x = self.dropout(x)
        x = x.view(x.size()[0] * x.size()[1], self.lstm_hidden_size)
        x = self.fc(x)
        return x, (h, c)

    def predict(self, char, dataset, device, previous_hidden_states, top_k=5):
        self.to(device)
        x = np.array([[dataset.char2int[char]]])
        x = dataset.one_hot_encode(x)

        inputs = torch.from_numpy(x).to(device)

        previous_hidden_states = tuple([each.data for each in previous_hidden_states])
        out, previous_hidden_states = self.forward(inputs, previous_hidden_states)

        p = F.softmax(out, dim=1).data
        p = p.cpu()

        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p / p.sum())
        return dataset.int2char[char], previous_hidden_states

    def init_hidden(self, sequences_per_batch):
        weight = next(self.parameters()).data
        return (
            weight.new(
                self.number_lstm_layers, sequences_per_batch, self.lstm_hidden_size
            )
            .zero_()
            .to(self.device),
            weight.new(
                self.number_lstm_layers, sequences_per_batch, self.lstm_hidden_size
            )
            .zero_()
            .to(self.device),
        )

