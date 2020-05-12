import torch
from torch.utils.data.dataset import Dataset
import numpy as np


def read_whole_file(input_path):
    with open(input_path, mode="r") as file:
        return file.read()


class CustomDatasetLoader(Dataset):
    def __init__(self, input_path, validation_split=0.1):
        self.data = read_whole_file(input_path)

        # Unique characters in the database.
        self.unique_characters = set(self.data)
        self.unique_characters_length = len(self.unique_characters)

        # Map int to character.
        self.int2char = {i: char for i, char in enumerate(self.unique_characters)}
        # Map character to int.
        self.char2int = {char: i for i, char in enumerate(self.unique_characters)}

        data_encoded = np.array(self.characters2int(self.data))
        split_index = int(len(data_encoded) * validation_split)
        self.train_data = data_encoded[split_index:]
        self.validation_data = data_encoded[:split_index]

    def get_validation_batches(self):
        pass

    def get_train_batches(self, batch_size=3, sequence_size=8):
        # Total size of all the batches, considering the sequence size.
        total_batches_size = batch_size * sequence_size
        # Number of batches that can be made.
        batches = len(self.train_data) // total_batches_size

        # Data resized to be used in batches.
        data_encoded = self.train_data[: batches * total_batches_size]
        data_encoded = self.train_data.reshape((batches, -1))

        for i in range(0, sequence_size, sequence_size):
            x = data_encoded[:, i : i + sequence_size]
            y = np.zeros_like(x)
            y[:, :-1] = x[:, 1:]

            # Remove last character, because it has no next element.
            x = x[:, :-1]
            y = y[:, :-1]

            x = self.one_hot_encode(x)
            yield torch.tensor(x).float().cuda(), torch.from_numpy(y).cuda()

    def characters2int(self, characters):
        return [self.char2int[c] for c in characters]

    def int2characters(self, characters):
        return [self.int2char[c] for c in characters]

    def one_hot_encode(self, characters):
        batches = characters.shape[0]
        sequence_size = characters.shape[1]
        encoded = np.zeros(
            [batches, sequence_size, self.unique_characters_length], dtype=int,
        )
        for i in range(batches):
            for j in range(sequence_size):
                encoded[i][j][characters[i][j]] = 1
        return encoded

    def one_hot_decode(self, characters):
        return [np.argmax(x) for x in characters]
