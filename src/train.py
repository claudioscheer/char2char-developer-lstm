import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


def train_model(model, dataset, show_loss_plot=False, n_epochs=1000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_over_epochs = []
    validation_loss_over_epochs = []

    for epoch in range(1, n_epochs + 1):
        hidden_states = model.init_hidden_states(3)
        for x, y in dataset.get_train_batches(batch_size=1):
            hidden_states = tuple([each.data for each in hidden_states])
            optimizer.zero_grad()
            output, _ = model(x, hidden_states)
            train_loss = criterion(output, y.view(-1).long())
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        # for batch_index, (x, y) in enumerate(validation_loader):
        #     hidden_states = model.init_hidden_states(1)
        #     output, _ = model(x, hidden_states)
        #     validation_loss = criterion(output, y.view(-1).long())

        train_loss_over_epochs.append(train_loss.item())
        # validation_loss_over_epochs.append(validation_loss.item())
        print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
        print("Loss: {:.4f}".format(train_loss.item()))

    if show_loss_plot:
        plt.plot(train_loss_over_epochs, label="Train loss")
        plt.plot(validation_loss_over_epochs, label="Validation loss")
        plt.legend()
        plt.title("Loss")
        plt.show()

    return model
