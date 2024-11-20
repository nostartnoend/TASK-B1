import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()  # 输出范围在 [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 实例化模型并移至设备
model = DenoisingAutoencoder().to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模型训练
num_epochs = 10
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(-1, 28 * 28).to(device)

        # 添加噪声
        noisy_img = img + 0.5 * torch.randn(img.size()).to(device)
        noisy_img = torch.clamp(noisy_img, 0., 1.)  # 限制范围

        # 向前传播
        optimizer.zero_grad()
        output = model(noisy_img)
        loss = criterion(output, img)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# 使用模型重建图像
model.eval()  # 切换到评估模式

with torch.no_grad():
    # 获取一批图像
    test_img, _ = next(iter(train_loader))
    test_img = test_img.view(-1, 28 * 28).to(device)

    # 添加噪声
    noisy_test_img = test_img + 0.5 * torch.randn(test_img.size()).to(device)
    noisy_test_img = torch.clamp(noisy_test_img, 0., 1.)

    # 重建图像
    reconstructed = model(noisy_test_img)


# 可视化
def show_images(original, noisy, reconstructed):
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(original.view(28, 28).cpu(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Noisy')
    plt.imshow(noisy.view(28, 28).cpu(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Reconstructed')
    plt.imshow(reconstructed.view(28, 28).cpu(), cmap='gray')
    plt.axis('off')

    plt.show()


# 显示图像
for i in range(3):  # 显示前 3 张图像
    show_images(test_img[i], noisy_test_img[i], reconstructed[i])