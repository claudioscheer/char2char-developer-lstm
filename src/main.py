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

batch_size = 1
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
validation_loader = DataLoader(latex_dataset, batch_size=1, sampler=validation_sampler)
test_loader = DataLoader(latex_dataset, batch_size=1, sampler=test_sampler)

model = LSTMModel(
    latex_dataset.unique_characters_length, latex_dataset.unique_characters_length
)
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Starting train process...")

train_loss_over_epochs = []
validation_loss_over_epochs = []

n_epochs = 100
for epoch in range(1, n_epochs + 1):
    for batch_index, (x, y) in enumerate(train_loader):
        hidden_states = model.init_hidden_states(batch_size)
        optimizer.zero_grad()
        output, _ = model(x, hidden_states)
        train_loss = criterion(output, y.view(-1).long())
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    for batch_index, (x, y) in enumerate(validation_loader):
        hidden_states = model.init_hidden_states(1)
        output, _ = model(x, hidden_states)
        validation_loss = criterion(output, y.view(-1).long())

    train_loss_over_epochs.append(train_loss.item())
    validation_loss_over_epochs.append(validation_loss.item())
    print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
    print("Loss: {:.4f}".format(train_loss.item()))


torch.save(model, "../model.pytorch")

model.eval()
with torch.no_grad():
    for batch_index, (x, y) in enumerate(test_loader):
        hidden_states = model.init_hidden_states(1)
        output, _ = model(x, hidden_states)
        test_loss = criterion(output, y.view(-1).long())
        print("Testing loss: {:.4f}".format(test_loss.item()))


plt.plot(train_loss_over_epochs, label="Train loss")
plt.plot(validation_loss_over_epochs, label="Validation loss")
plt.legend()
plt.title("Loss")
plt.show()
