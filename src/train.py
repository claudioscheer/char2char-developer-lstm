import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


def train_model(model, dataset, show_loss_plot=False, n_epochs=16, sequence_size=16):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loss_over_epochs = []
    validation_loss_over_epochs = []

    for epoch in range(1, n_epochs + 1):
        hidden_states = model.init_hidden_states(1)
        optimizer.zero_grad()

        x, y = dataset.get_batch(sequence_size=sequence_size)
        train_loss = 0

        for i in range(sequence_size):
            output, hidden_states = model(x[i], hidden_states)
            train_loss += criterion(output, y[i].unsqueeze(0))

        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # x_validation, y_validation = dataset.get_batch(batch_size=1, sequence_size=1)
        # hidden_states = model.init_hidden_states(1)
        # output_validation, _ = model(x_validation, hidden_states)
        # validation_loss = criterion(output_validation, y_validation.view(-1).long())

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
