import torch
import torch.nn as nn
import random


class Char2Char(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Char2Char, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.lstm_hidden_size == decoder.lstm_hidden_size
        ), "Hidden dimensions of encoder and decoder must be equal."
        assert (
            encoder.number_lstm_layers == decoder.number_lstm_layers
        ), "Encoder and decoder must have the same number of layers."

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        batch_size = y.shape[1]
        y_len = y.shape[0]
        y_vocab_size = self.decoder.output_size

        outputs = torch.zeros(y_len, batch_size, y_vocab_size).to(self.device)

        hidden_states = self.encoder(x)
        y_input = y[0, :]

        for next_y in range(1, y_len):
            output, hidden_states = self.decoder(y_input, hidden_states)
            outputs[next_y] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            y_input = y[next_y] if teacher_force else top1

        return outputs
