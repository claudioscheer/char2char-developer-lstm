import numpy as np
import torch
import torch.nn as nn
from lib.dataset import RandomDatasetLoader
from lib.model import LSTMModel
from train import train_model


torch.autograd.set_detect_anomaly(True)
dataset = RandomDatasetLoader("../dataset/latex.txt")

model = LSTMModel(dataset.unique_characters_length, dataset.unique_characters_length)
model.cuda()

print("Starting train process...")

model = train_model(
    model, dataset, show_loss_plot=True, n_epochs=1000, sequence_size=256
)
torch.save(model, "../model.pytorch")
