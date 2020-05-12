import torch
import torch.nn as nn
from torch.autograd import Variable
from lib.dataset import RandomDatasetLoader
from lib.model import LSTMModel
import numpy as np


dataset = RandomDatasetLoader("../dataset/alphabet.txt")

model = torch.load("../model.pytorch")
model.cpu()
model.eval()


# def encode_character(character):
#     character = dataset.characters2int(character)
#     character = torch.tensor(character)
#     character.unsqueeze_(axis=0)
#     character = dataset.one_hot_encode(character)
#     character = torch.from_numpy(character).float()
#     return character


# def get_predicted_character(output):
#     probability = nn.functional.softmax(output[-1], dim=0).data
#     output = torch.max(probability, dim=0)[1].item()
#     output = dataset.int2char[output]
#     return output


# def predict(model, prediction_length, start_text):
#     characters = [x for x in start_text]
#     size_prediction = prediction_length - len(characters)

#     for character in characters:
#         output, previous_hidden_states = model(
#             encode_character(character), previous_hidden_states
#         )

#     for x in range(size_prediction):
#         output, previous_hidden_states = model(
#             encode_character(characters), previous_hidden_states
#         )
#         output = get_predicted_character(output)
#         characters.append(output)

#     return characters


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
        output, hidden = model(inp, previous_hidden_states)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = dataset.int2char[top_i.item()]
        predicted += predicted_char
        inp = Variable(torch.tensor(dataset.characters2int(predicted_char)))

    return predicted


with torch.no_grad():
    prediction = evaluate(model, "abcdefghij", 12)
    print("".join(prediction))
