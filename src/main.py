import os
import numpy as np
import torch
import torch.nn as nn
from lib.dataset import RandomDatasetLoader
from lib.model import LSTMModel
from train import train_model


file_path = os.path.dirname(os.path.abspath(__file__))
dataset = RandomDatasetLoader(os.path.join(file_path, "../dataset/source-code.txt"))

model = LSTMModel(dataset.unique_characters_length, dataset.unique_characters_length)
model.cuda()

print("Starting train process...")

model = train_model(
    model, dataset, show_loss_plot=True, n_epochs=128, sequence_size=256
)
torch.save(model, "../model.pytorch")
