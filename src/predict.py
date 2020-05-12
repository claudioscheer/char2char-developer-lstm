import torch
import torch.nn as nn
from lib import dataset
from lib.model import LSTMModel


latex_dataset = dataset.CustomDatasetLoader("../dataset/latex.txt")
dataset_size = len(latex_dataset)

model = torch.load("../model.pytorch")
model.cpu()
model.eval()

x, y = latex_dataset[0]
x = x.cpu()
y = y.cpu()
x.unsqueeze_(axis=0)
output, _ = model(x)
print(output)
print(y)


# def encode_character(characters):
#     characters = latex_dataset.characters2int(characters)
#     characters = latex_dataset.one_hot_encode(characters, len(characters))
#     characters = torch.tensor(characters).float()
#     characters.unsqueeze_(axis=0)
#     print(characters.shape)
#     return characters


# def get_predicted_character(output):
#     probability = nn.functional.softmax(output[-1], dim=0).data
#     output = torch.max(probability, dim=0)[1].item()
#     output = latex_dataset.int2char[output]
#     return output


# def predict(model, prediction_length, start_text):
#     characters = [x for x in start_text]
#     size_prediction = prediction_length - len(characters)

#     for x in range(size_prediction):
#         output, hidden = model(encode_character(characters))
#         output = get_predicted_character(output)
#         characters.append(output)

#     return characters


# with torch.no_grad():
#     prediction = predict(model, 2, "a")
#     print("".join(prediction))
