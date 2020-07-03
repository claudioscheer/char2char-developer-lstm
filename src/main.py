import os
import numpy as np
import torch
import torch.nn as nn
from lib.dataset import RandomDatasetLoader
from lib.char2char import Char2Char
from lib.encoder import Encoder
from lib.decoder import Decoder
from train import train_model


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for _, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


file_path = os.path.dirname(os.path.abspath(__file__))
dataset = RandomDatasetLoader(os.path.join(file_path, "../dataset/source-code.txt"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters.
dictionary_size = dataset.dictionary_size
embedding_dim = 128
lstm_hidden_size = 256
number_lstm_layers = 2
dropout_probability = 0.5
output_size = dataset.dictionary_size

encoder = Encoder(
    dictionary_size,
    embedding_dim,
    lstm_hidden_size,
    number_lstm_layers,
    dropout_probability,
)
decoder = Decoder(
    lstm_hidden_size,
    embedding_dim,
    lstm_hidden_size,
    number_lstm_layers,
    dropout_probability,
    output_size,
)
model = Char2Char(encoder, decoder, device).to(device)
model.apply(init_weights)

print(model)

model = train_model(
    model, dataset, device, show_loss_plot=True, n_epochs=128, sequence_size=256
)
torch.save(model, "../model.pytorch")
