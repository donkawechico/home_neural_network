import torch
from torch import tensor
import numpy as np
from sklearn.model_selection import train_test_split
from .data_fetcher import fetch_data
from .util import merge_data


class Net(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 10)
        self.fc2 = torch.nn.Linear(10, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(hass, entity):
    """Handle the service call."""
    # Fetch the merged input and label data frame
    input_entity_ids = entity.attributes["inputs"]
    label_entity_ids = entity.attributes["labels"]
    all_entity_ids = input_entity_ids + label_entity_ids
    all_data = fetch_data(hass, all_entity_ids)
    all_df = merge_data(all_data)

    # Split the data into training and testing sets
    X = all_df[input_entity_ids].values
    y = all_df[label_entity_ids].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert the data to PyTorch tensors
    X_train_tensor = tensor(X_train.astype(np.float32))
    y_train_tensor = tensor(y_train.astype(np.float32))
    X_test_tensor = tensor(X_test.astype(np.float32))
    y_test_tensor = tensor(y_test.astype(np.float32))

    net = Net(len(input_entity_ids), len(label_entity_ids))

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    # Train the model
    num_epochs = 10000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = net(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(
                "Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item())
            )

    # Evaluate the model
    with torch.no_grad():
        outputs = net(X_test_tensor)
        test_loss = criterion(outputs, y_test_tensor)
        print("Test Loss: {:.4f}".format(test_loss.item()))

    return net
