import torch
import torch.nn as nn
import torch.nn.functional as F

class CLDNN1D(nn.Module):
    def __init__(self, num_classes=22, kernel_size=8, seq_length=2048):
        super(CLDNN1D, self).__init__()
        # Assuming input_size is the length of the sequences, and input has 2 channels
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=8)
        # Define LSTM and Dropout layers
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.4)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.4)
        # Flatten layer will be applied dynamically in the forward method
        conv1d_output_length = (seq_length - (kernel_size - 1) - 1) + 1
        pooled_output_length = conv1d_output_length // 2
        # Define the fully connected layer with the correct input size
        self.fc = nn.Linear(64 * pooled_output_length, num_classes)

    def forward(self, x):
        # Convert (batch, seq, feature) to (batch, feature, seq) for Conv1D
        x = x.transpose(1, 2)
        x = F.relu(self.conv1d(x))
        x = F.max_pool1d(x, 2)
        # Convert back to (batch, seq, feature) for LSTM
        x = x.transpose(1, 2)
        # X(32,508,64)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        # Flatten the output for the fully connected layer
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
