import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from lib import dataset
from lib.model import LSTMModel


latex_dataset = dataset.CustomDatasetLoader("../dataset/latex.txt")
dataset_size = len(latex_dataset)
# -1 to remove the last character from the dataset.
dataset_indices = list(range(dataset_size - 1))

batch_size = 16
# Shuffle dataset indices. I think this is unnecessary.
np.random.shuffle(dataset_indices)

train_sampler = SubsetRandomSampler(dataset_indices)
train_loader = DataLoader(latex_dataset, batch_size=batch_size, sampler=train_sampler)

model = LSTMModel(
    latex_dataset.unique_characters_length, latex_dataset.unique_characters_length
)
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

n_epochs = 20
for epoch in range(1, n_epochs + 1):
    for batch_index, (x, y) in enumerate(train_loader):
        # Add axis 1 dimension. See docs/draws/input-output-sizes.ora.
        x.unsqueeze_(axis=1)
        optimizer.zero_grad()

        output, hidden_state = model(x)
        loss = criterion(output, y.view(-1).long())
        loss.backward()
        optimizer.step()

    print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
    print("Loss: {:.4f}".format(loss.item()))
