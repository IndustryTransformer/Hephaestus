# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)
print(f"Using device: {device}")


# Define the CNN model
class HandwritingRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Define the pooling and dropout layers
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Define the fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout2(x)

        # Reshape the output for the fully connected layers
        x = x.view(-1, 32 * 7 * 7)

        # Pass the output through the fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        # Return the final output
        return x


# Load the MNIST dataset
train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
test_dataset = MNIST("./data", train=False, download=True, transform=ToTensor())

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the model
model = HandwritingRecognitionModel().to(device)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model for 10 epochs
for epoch in range(10):
    # Set the model to training mode
    model.train()

    # Iterate over the training data
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Pass the input through the model
        outputs = model(images)

        # Compute the loss
        loss = loss_fn(outputs, labels)

        # Backpropagate the error
        loss.backward()

        # Update the model parameters
        optimizer.step()

    # Set the model to evaluation mode
    model.eval()

    # Evaluate the model on the validation set
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # Pass the input through the model
            outputs = model(images)

            # Get the predicted labels
            _, predicted = torch.max(outputs.data, 1)

            # Update the total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum()

        # Print the accuracy
        print(f"Epoch {epoch + 1}: Accuracy = {100 * correct / total:.2f}%")


# %%
