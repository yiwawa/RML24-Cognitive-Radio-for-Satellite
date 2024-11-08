import torch.nn as nn
import torch

class CNN4(nn.Module):
    def __init__(self, num_classes=22, init_weights=False):
        super(CNN4, self).__init__()

        # Convolution layer 1 (（w - f + 2 * p）/ s ) + 1
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=256, kernel_size=8)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_drop = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=8)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_drop = nn.Dropout(0.5)


        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3_drop = nn.Dropout(0.5)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4_drop = nn.Dropout(0.5)

        # Fully-Connected layer 2

        self.fc1 = nn.Linear(7744, 256)  ##TODO
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)

        # Fully-Connected layer 2
        self.fc2 = nn.Linear(256,num_classes)



    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv1_drop(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv2_drop(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.conv3_drop(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.conv4_drop(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc2(x)

        return x

