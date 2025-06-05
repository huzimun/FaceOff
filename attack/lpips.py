import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AlexNetFeatures(nn.Module):
    def __init__(self, device='cpu'):
        super(AlexNetFeatures, self).__init__()
        alexnet = models.alexnet(pretrained=True).to(device)
        self.features = alexnet.features
        self.device = device

        # 选择一些中间层作为特征输出
        self.layers = [2, 5, 8, 10, 12]  # Conv1, Conv2, Conv3, Conv4, Conv5

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layers:
                outputs.append(x)
                # print(f"Layer {i} output shape: {x.shape}")  # Debug information
        return outputs

class LPIPSLoss(nn.Module):
    def __init__(self, device='cpu', dtype=torch.float):
        super(LPIPSLoss, self).__init__()
        self.alex_features = AlexNetFeatures(device=device)
        for param in self.alex_features.parameters():
            param.requires_grad = False

        # 创建测试输入以获取正确的通道数
        test_input = torch.randn(1, 3, 224, 224).to(device)
        test_outputs = self.alex_features(test_input)
        
        # 根据实际特征层输出创建线性层
        self.lin_layers = nn.ModuleList([
            nn.Conv2d(out.shape[1], 1, kernel_size=1, bias=False).to(device)
            for out in test_outputs
        ])
        self.device = device
        self.dtype = dtype

    def forward(self, x, y):
        x_feats = self.alex_features(x.to(self.device).to(self.dtype))
        y_feats = self.alex_features(y.to(self.device).to(self.dtype))

        loss = 0
        for xf, yf, lin in zip(x_feats, y_feats, self.lin_layers):
            diff = F.normalize(xf, p=2, dim=1) - F.normalize(yf, p=2, dim=1)
            loss += lin(diff ** 2).mean()

        return loss

# 使用示例：
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = LPIPSLoss(device=device)

    # 创建两个随机图像张量，模拟输入
    x = torch.rand(1, 3, 224, 224).to(device)  # 图像 1
    y = torch.rand(1, 3, 224, 224).to(device)  # 图像 2

    loss = criterion(x, y)
    print(f'LPIPS Loss: {loss.item():.4f}')
