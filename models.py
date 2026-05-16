import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class ECGCNN(nn.Module):
    def __init__(self, num_classes):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
        # 187 -> 183 (conv1) -> 91 (pool) -> 87 (conv2) -> 43 (pool) -> 39 (conv3) -> 19 (pool)
        self.fc1 = nn.Linear(128 * 19, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # [B, 187] -> [B, 1, 187]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
