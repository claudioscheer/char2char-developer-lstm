import torch
import torch.nn as nn
from lib import dataset
from lib.model import LSTMModel
import numpy as np


latex_dataset = dataset.CustomDatasetLoader("../dataset/latex.txt")

model = torch.load("../model.pytorch")
model.cpu()
model.eval()


def encode_character(character):
    character = latex_dataset.characters2int(character)
    character = torch.tensor(character)
    character.unsqueeze_(axis=0)
    character = latex_dataset.one_hot_encode(character)
    character = torch.from_numpy(character).float()
    print(character.shape)
    return character


def get_predicted_character(output):
    probability = nn.functional.softmax(output[-1], dim=0).data
    output = torch.max(probability, dim=0)[1].item()
    output = latex_dataset.int2char[output]
    return output


def predict(model, prediction_length, start_text):
    characters = [x for x in start_text]
    size_prediction = prediction_length - len(characters)
    previous_hidden_states = model.init_hidden_states(1, False)

    for character in characters:
        output, previous_hidden_states = model(
            encode_character(character), previous_hidden_states
        )
        print(get_predicted_character(output))

    # for x in range(size_prediction):
    #     output, previous_hidden_states = model(
    #         encode_character(characters), previous_hidden_states
    #     )
    #     output = get_predicted_character(output)
    #     characters.append(output)

    return characters


with torch.no_grad():
    prediction = predict(model, 2, "ab")
    print("".join(prediction))
