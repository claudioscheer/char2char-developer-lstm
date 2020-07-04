import os
import numpy as np
import torch
import torch.nn as nn
from lib.dataset import DatasetLoader
from lib.char2char import Char2Char
from train import train_model
from predict import sample


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = os.path.dirname(os.path.abspath(__file__))
dataset = DatasetLoader(os.path.join(file_path, "../dataset/source-code.txt"))

# Define hyperparameters.
dictionary_size = dataset.dictionary_size
embedding_size = 128
lstm_hidden_size = 512
number_lstm_layers = 2
dropout_probability = 0.5
output_size = dataset.dictionary_size
sequences_per_batch = 16
sequence_length = 16
learning_rate = 1e-3
number_epochs = 32

model = Char2Char(
    dictionary_size,
    embedding_size,
    lstm_hidden_size,
    number_lstm_layers,
    dropout_probability,
    output_size,
    device,
).to(device)

print(model)

model = train_model(
    model,
    dataset,
    device,
    sequences_per_batch=sequences_per_batch,
    sequence_length=sequence_length,
    epochs=number_epochs,
    learning_rate=learning_rate,
    show_loss_plot=True,
)

output = sample(model, dataset, device, 1000, "int adxl_decode(")
print(output)
