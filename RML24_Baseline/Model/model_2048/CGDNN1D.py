import torch
import torch.nn as nn
import torch.optim as optim

class CGDNN_1D(nn.Module):
    def __init__(self, num_classes=22):
        super(CGDNN_1D, self).__init__()
        self.conv1d1 = nn.Conv1d(in_channels=2, out_channels=80, kernel_size=7)
        self.conv1d2 = nn.Conv1d(in_channels=80, out_channels=80, kernel_size=7)
        self.conv1d3 = nn.Conv1d(in_channels=80, out_channels=80, kernel_size=7)
        self.pool1d = nn.MaxPool1d(kernel_size=2)
        self.gru1 = nn.GRU(80, 40, batch_first=True)
        self.dropout1 = nn.Dropout(0.4)
        self.gru2 = nn.GRU(40, 20, batch_first=True)
        self.dropout2 = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.avgpool1d = nn.AvgPool1d(kernel_size=2)
        self.fc1 = nn.Linear(2500, 640)  # TODO
        self.fc2 = nn.Linear(640, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1d1(x))
        x = self.pool1d(x)
        x = torch.relu(self.conv1d2(x))
        x = self.pool1d(x)
        x = torch.relu(self.conv1d3(x))
        x = self.pool1d(x)

        x = x.transpose(1, 2)
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        x = x.transpose(1, 2)
        x = self.avgpool1d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x

