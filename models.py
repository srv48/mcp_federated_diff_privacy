import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)
        self.act = nn.ReLU6(inplace=True)  # Efficient ReLU
        self.dropout = nn.Dropout(0.99)     # Lightweight regularization

    def forward(self, x):
        x = self.pool(self.act(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class BigCNN(nn.Module):
    def __init__(self):
        super(BigCNN, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)
        self.act = nn.ReLU(inplace=True)  # Standard ReLU
        # No dropout — heavier training

    def forward(self, x):
        x = self.pool(self.act(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x