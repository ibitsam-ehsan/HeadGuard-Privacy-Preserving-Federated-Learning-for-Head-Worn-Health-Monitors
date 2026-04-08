import torch
import torch.nn as nn
from headguard import HeadGuard

# Simple model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, 5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Dummy data
X = torch.randn(100, 1, 640)
y = torch.randint(0, 2, (100,))

# Train with HeadGuard
hg = HeadGuard(epsilon=4.0)
model = SimpleCNN()
train_loader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=32)

hg.train(model, train_loader, epochs=5)
print("Training complete")
