import unidecode
import string
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import random


def read_whole_file(input_path):
    with open(input_path, mode="r") as file:
        return file.read()


class DatasetLoader(Dataset):
    def __init__(self, input_path, validation_split=0.0):
        self.data = read_whole_file(input_path)
        self.dictionary_size = len(string.printable)

        self.int2char = {i: char for i, char in enumerate(string.printable)}
        self.char2int = {char: i for i, char in enumerate(string.printable)}
        self.data_encoded = np.array(self.characters2int(self.data))

        validation_index = int(len(self.data_encoded) * (1 - validation_split))
        train_data, validation_data = (
            self.data_encoded[:validation_index],
            self.data_encoded[validation_index:],
        )
        self.train_data = train_data
        self.validation_data = validation_data

    def get_train_batch(self, sequences_per_batch, sequence_length):
        batch_size = sequences_per_batch * sequence_length
        number_batches = len(self.train_data) // batch_size
        data = self.train_data[: number_batches * batch_size]
        data = data.reshape((sequences_per_batch, -1))

        for n in range(0, data.shape[1], sequence_length):
            x = data[:, n : n + sequence_length]
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], data[:, n + sequence_length]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], data[:, 0]
            yield x, y

    def characters2int(self, characters):
        return [self.char2int[c] for c in characters]
