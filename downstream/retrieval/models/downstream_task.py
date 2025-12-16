import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class LinearProbe(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.classification_head = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.classification_head(x)  

class SimpleSelfAttentionNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleSelfAttentionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = x.unsqueeze(0)  
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(0)  
        x = self.fc3(x)
        return x
    
