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
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1]
])

# 加载 CIFAR-10 数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # CIFAR-10
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 输出：32 x 16 x 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 输出：64 x 8 x 8
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 输出：128 x 4 x 4
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 输出：256 x 2 x 2
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 输出：128 x 4 x 4
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 输出：64 x 8 x 8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 输出：32 x 16 x 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 输出：3 x 32 x 32
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
        img = img.to(device)

        # 添加噪声
        noisy_img = img + 0.5 * torch.randn(img.size()).to(device)
        noisy_img = torch.clamp(noisy_img, -1., 1.)  # 限制范围

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
    test_img = test_img.to(device)

    # 添加噪声
    noisy_test_img = test_img + 0.5 * torch.randn(test_img.size()).to(device)
    noisy_test_img = torch.clamp(noisy_test_img, -1., 1.)

    # 重建图像
    reconstructed = model(noisy_test_img)


# 可视化
def show_images(original, noisy, reconstructed):
    plt.figure(figsize=(12, 4))

    # 显示原始图像
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow((original.permute(1, 2, 0).cpu().numpy() + 1) / 2)  # 反归一化到 [0, 1]
    plt.axis('off')

    # 显示带噪声的图像
    plt.subplot(1, 3, 2)
    plt.title('Noisy')
    plt.imshow((noisy.permute(1, 2, 0).cpu().numpy() + 1) / 2)  # 反归一化到 [0, 1]
    plt.axis('off')

    # 显示重建图像
    plt.subplot(1, 3, 3)
    plt.title('Reconstructed')
    plt.imshow((reconstructed.permute(1, 2, 0).cpu().numpy() + 1) / 2)  # 反归一化到 [0, 1]
    plt.axis('off')

    plt.show()


# 显示图像
for i in range(3):  # 显示前 3 张图像
    show_images(test_img[i], noisy_test_img[i], reconstructed[i])