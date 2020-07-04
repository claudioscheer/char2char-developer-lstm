import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def train_model(
    model,
    dataset,
    device,
    sequences_per_batch,
    sequence_length,
    epochs,
    learning_rate,
    show_loss_plot=False,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_over_epochs = []
    validation_loss_over_epochs = []

    for epoch in range(1, epochs + 1):
        hidden_states = model.init_hidden(sequences_per_batch)
        for x, y in dataset.get_train_batch(sequences_per_batch, sequence_length):
            x = torch.tensor(x).to(device)
            y = torch.from_numpy(y).to(device)

            hidden_states = tuple([each.data for each in hidden_states])
            model.zero_grad()
            output, hidden_states = model.forward(x, hidden_states)
            loss = criterion(
                output, y.view(sequences_per_batch * sequence_length).long(),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_loss_over_epochs.append(loss.item())
        print(f"Epoch {epoch}/{epochs}...")

    if show_loss_plot:
        plt.plot(train_loss_over_epochs, label="Train loss")
        plt.plot(validation_loss_over_epochs, label="Validation loss")
        plt.legend()
        plt.title("Loss")
        plt.show()

    return model
