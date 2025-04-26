import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, in_channel=3, num_classes=1, image_size=120):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Compute flattened feature size dynamically
        self.feature_size = self._get_flattened_size(in_channel, image_size)

        self.fc1 = nn.Linear(self.feature_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, num_classes)

        # Use sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()

    def _get_flattened_size(self, in_channels, image_size):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            x = self.pool1(F.relu(self.conv1(dummy)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.fc2(x)

        return self.sigmoid(x)
