import unidecode
import string
import torch
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import numpy as np
import random


def read_whole_file(input_path):
    with open(input_path, mode="r") as file:
        return unidecode.unidecode(file.read())


class RandomDatasetLoader(Dataset):
    def __init__(self, input_path, validation_split=0.1):
        self.data = read_whole_file(input_path)
        self.data_len = len(self.data)
        self.dictionary_size = len(string.printable)

        # Map int to character.
        self.int2char = {i: char for i, char in enumerate(string.printable)}
        # Map character to int.
        self.char2int = {char: i for i, char in enumerate(string.printable)}

        self.data_encoded = self.characters2int(self.data)

    def __len__(self):
        return len(self.data)

    def get_random_chunk(self, length):
        start_index = random.randint(0, self.data_len - length)
        end_index = start_index + length
        return self.data_encoded[start_index:end_index]

    def get_batch(self, sequence_size=16):
        # Required because 1 element is removed from x and y.
        sequence_size += 1

        chunk = self.get_random_chunk(sequence_size)
        # Ensure that random chunk has the sequence size.
        while len(chunk) != sequence_size:
            chunk = self.get_random_chunk(sequence_size)
        # Remove last character.
        x = chunk[:-1]
        # Remove first character.
        y = chunk[1:]

        x = torch.tensor(x).cuda()
        y = torch.tensor(y).cuda()
        return Variable(x), Variable(y)

    def characters2int(self, characters):
        return [self.char2int[c] for c in characters]

    def int2characters(self, characters):
        return [self.int2char[c] for c in characters]
