import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from lib.dataset import RandomDatasetLoader
from lib.model import LSTMModel
import numpy as np


file_path = os.path.dirname(os.path.abspath(__file__))
dataset = RandomDatasetLoader(os.path.join(file_path, "../dataset/source-code.txt"))

model = torch.load("../model.pytorch")
model.cpu()
model.eval()


def evaluate(model, start_text, prediction_length, temperature=0.8):
    previous_hidden_states = model.init_hidden_states(1, False)
    prime_input = Variable(torch.tensor(dataset.characters2int(start_text)))
    predicted = start_text

    # Use priming string to "build up" hidden state.
    for p in range(len(start_text) - 1):
        _, previous_hidden_states = model(prime_input[p], previous_hidden_states)
    inp = prime_input[-1]

    size_prediction = prediction_length - len(start_text)

    for p in range(size_prediction):
        output, previous_hidden_states = model(inp, previous_hidden_states)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = dataset.int2char[top_i.item()]
        predicted += predicted_char
        inp = Variable(torch.tensor(dataset.characters2int(predicted_char)))

    return predicted


with torch.no_grad():
    prediction = evaluate(model, "static int", 5000)
    print("".join(prediction))
