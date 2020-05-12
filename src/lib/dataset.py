import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class CustomDatasetLoader(Dataset):
    def __init__(self, input_path):
        self.sequence_size = 16
        self.input_path = input_path
        self.data = self.read_whole_file()
        self.data_len = len(self.data)

        # Unique characters in the database.
        self.unique_characters = set(self.data)
        self.unique_characters_length = len(self.unique_characters)

        # Map int to character.
        self.int2char = {i: char for i, char in enumerate(self.unique_characters)}
        # Map character to int.
        self.char2int = {char: i for i, char in enumerate(self.unique_characters)}

    def __getitem__(self, index):
        x, y = self.get_dataset_tuple(index)
        # Map text to int.
        x = self.characters2int(x)
        # One-hot encode x.
        x = self.one_hot_encode(x, self.sequence_size)
        x = torch.tensor(x).float().cuda()

        # Map text to int.
        y = self.characters2int(y)
        # y = self.one_hot_encode(y, self.sequence_size)
        y = torch.tensor(y).cuda()
        return x, y

    def __len__(self):
        return self.data_len - self.sequence_size

    def get_dataset_tuple(self, index):
        x = []
        y = []
        for j in range(0, self.sequence_size):
            x.append(self.data[index + j])
            y.append(self.data[index + j + 1])
        return x, y

    def read_whole_file(self):
        with open(self.input_path, mode="r") as file:
            return file.read()

    def characters2int(self, characters):
        return [self.char2int[c] for c in characters]

    def int2characters(self, characters):
        return [self.int2char[c] for c in characters]

    def one_hot_encode(self, characters, sequence_size):
        encoded = np.zeros([sequence_size, self.unique_characters_length], dtype=int)
        for i, x in enumerate(characters):
            encoded[i][x] = 1
        return encoded

    def one_hot_decode(self, characters):
        return [np.argmax(x) for x in characters]
