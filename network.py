from torch import nn
import torch


class Network(nn.Module):
    def __init__(self, size=128, num_channels=3, batch_size=64, out_channels=8):
        super(Network, self).__init__()
        self.batch_size = batch_size
        self.size = size
        self.convolutional = nn.Sequential(
            # First convolutional layer: 1 input channel (for grayscale MNIST), 8 output channels
            nn.Conv2d(in_channels=num_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # Second convolutional layer: 8 input channels, 8 output channels
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces width and height by half
            # Third convolutional layer: 8 input channels, 16 output channels
            nn.Conv2d(in_channels=out_channels, out_channels=2*out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fourth convolutional layer: 16 input channels, 16 output channels
            nn.Conv2d(in_channels=2*out_channels, out_channels=2*out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.fully_connected = nn.Sequential(
                nn.Linear(16384, 500),
                nn.ReLU(),
                nn.Linear(500, 250),
                nn.ReLU(),
                nn.Linear(250, 1),
                nn.Sigmoid())

    def forward(self, x):
        batch_size = x.size(0)
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(batch_size, -1)
        x = self.fully_connected(x)
        return x

    def loss(self, prediction, label, reduction='mean'):
        prediction = torch.squeeze(prediction)
        label = torch.squeeze(label)
        loss_val = nn.BCELoss(reduction=reduction)(prediction, label.float())
        return loss_val
