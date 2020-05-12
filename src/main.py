import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from lib import dataset
from lib.model import LSTMModel
import matplotlib.pyplot as plt


latex_dataset = dataset.CustomDatasetLoader("../dataset/latex.txt")
dataset_size = len(latex_dataset)
test_split = int(np.floor(0.15 * dataset_size))  # 15%
validation_split = int(np.floor(0.15 * dataset_size))  # 15%
dataset_indices = list(range(dataset_size))

batch_size = 16
# Shuffle dataset indices. I think this is unnecessary.
np.random.shuffle(dataset_indices)

train_indices, validation_indices, test_indices = (
    dataset_indices[test_split + validation_split :],
    dataset_indices[test_split : test_split + validation_split],
    dataset_indices[:test_split],
)

train_sampler = SubsetRandomSampler(dataset_indices)
validation_sampler = SubsetRandomSampler(validation_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(latex_dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(
    latex_dataset, batch_size=batch_size, sampler=validation_sampler
)
test_loader = DataLoader(latex_dataset, batch_size=batch_size, sampler=test_sampler)

model = LSTMModel(
    latex_dataset.unique_characters_length, latex_dataset.unique_characters_length
)
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Starting train process...")

loss_over_epochs = []

n_epochs = 1000
for epoch in range(1, n_epochs + 1):
    for batch_index, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        output, hidden_state = model(x)
        loss = criterion(output, y.view(-1).long())
        loss.backward()
        optimizer.step()

    print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
    print("Loss: {:.4f}".format(loss.item()))
    loss_over_epochs.append(loss.item())

torch.save(model, "../model.pytorch")

plt.plot(loss_over_epochs)
plt.show()
