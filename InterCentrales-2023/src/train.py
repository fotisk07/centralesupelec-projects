import os

import torch
from torch.nn import L1Loss
from torch.optim import Adam

from .data import get_loader
from .Unet import get_model


def train(config, tracker):
    """ Uses the config to load the data and a model, then trains the model and logs the results using the tracker. """

    print("Starting training.")
    loader = get_loader(config)
    number_of_batches = config.number_of_batches if len(
        loader) > config.number_of_batches > -1 else len(loader)
    model = get_model(config)
    loss_function = L1Loss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.number_of_epochs):
        losses = []

        for i, data in enumerate(loader):

            images = data["s2"].permute(0, 3, 1, 2).float()

            input1 = torch.unsqueeze(images[:, 0, :, :], 1)
            input2 = torch.unsqueeze(images[:, 1, :, :], 1)
            if i == number_of_batches:
                break
            optimizer.zero_grad()
            outputs = model(input1, input2)
            loss = loss_function(outputs, data["label"])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % 2 == 1 or i+1 == number_of_batches:
                print(f"Performed batch {i+1}/{number_of_batches}")

        tracker.log_scalar("loss", total_loss := sum(
            losses)/len(losses), step=epoch)
        print(
            f"Epoch {epoch+1}/{config.number_of_epochs} : loss = {total_loss}.")
        torch.save(model.state_dict(), os.path.join(
            config.save_weights_under, f"epoch_{epoch}"))
