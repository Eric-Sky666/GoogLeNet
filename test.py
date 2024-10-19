import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# 定义简化版的 GoogleNet Inception 模块
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        output = torch.cat((b1, b2, b3, b4), 1)
        return output


# 定义简化版的 GoogleNet（仅需要定义以便加载模型）
class GoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet, self).__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)  # 修改这里，确保输出通道数为480

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.4)
        # 修改这里以匹配b3的输出通道数和maxpool后的空间维度
        self.fc = nn.Linear(480 * 3 * 3, num_classes)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.a3(x)
        x = self.b3(x)  # 确保b3的输出通道数为480
        x = self.maxpool(x)  # 假设输出空间维度为2x2（需要验证）
        x = x.view(x.size(0), -1)  # 展平为全连接层的输入
        x = self.fc(x)
        return x


# 数据预处理（与训练时相同，以确保输入一致）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载测试数据集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
print(f'The test data has been loaded')
# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GoogleNet(num_classes=10).to(device)
model.load_state_dict(torch.load("googlenet.pth"))
model.eval()
print(f'The model has been loaded')
# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
