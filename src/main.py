import numpy as np
import torch
import torch.nn as nn
from lib.dataset import CustomDatasetLoader
from lib.model import LSTMModel
from train import train_model


dataset = CustomDatasetLoader("../dataset/latex.txt")

model = LSTMModel(dataset.unique_characters_length, dataset.unique_characters_length)
model.cuda()

print("Starting train process...")

model = train_model(model, dataset, n_epochs=3000)
torch.save(model, "../model.pytorch")
