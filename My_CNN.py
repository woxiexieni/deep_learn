import torch 
import torch.nn as nn

class My_Cnn(nn.Module):
    def __init__(self):
        super(My_Cnn, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # 调整输入的形状
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
model = My_Cnn()
print(model)