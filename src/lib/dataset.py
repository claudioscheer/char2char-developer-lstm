import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class CustomDatasetLoader(Dataset):
    def __init__(self, input_path):
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
        x = self.char2int[x]
        # One-hot encode x.
        x = self.one_hot_encode(x)
        x = torch.tensor(x).float().cuda()

        # Map text to int.
        y = self.char2int[y]
        y = torch.tensor(y).cuda()
        return x, y

    def __len__(self):
        return self.data_len

    def get_dataset_tuple(self, index):
        x = self.data[index]
        y = self.data[index + 1]
        return x, y

    def read_whole_file(self):
        with open(self.input_path, mode="r") as file:
            return file.read()

    # def text2int(self, text):
    #     """
    #         Convert text to an array of integers.
    #     """
    #     return [self.char2int[c] for c in text]

    # def int2text(self, text):
    #     """
    #         Convert an array of integers to text.
    #     """
    #     return "".join([self.int2char[c] for c in text])

    def one_hot_encode(self, character):
        encoded = np.zeros([self.unique_characters_length], dtype=int)
        encoded[character] = 1
        return encoded

    def one_hot_decode(self, sequence):
        """
            sequence: PyTorch tensor.
        """
        return [np.argmax(x) for x in sequence.numpy()]
