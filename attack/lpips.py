import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        # import pdb; pdb.set_trace()
        return (inp - self.shift.to(inp.device)) / self.scale.to(inp.device)

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
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
        self.scaling_layer = ScalingLayer()
        self.chns = [64,192,384,256,256]
        self.L = len(self.chns)
        
        lpips = True
        use_dropout=True
        self.lpips = lpips
        self.spatial = False
        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            self.lins = nn.ModuleList(self.lins)

    def forward(self, x, y):
        normalize = True
        import pdb; pdb.set_trace()
        if normalize: # turn on this flag if input is [0,255] so it can be adjusted to [-1, +1]
            in0 = 2 * (x / 255.0) - 1
            in1 = 2 * (y / 255.0) - 1
        import pdb; pdb.set_trace()
        # v0.0 - original release had a bug, where input was not scaled
        # in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1))
        outs0, outs1 = self.alex_features.forward(in0_input), self.alex_features.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]
        
        return val

# 使用示例：
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = LPIPSLoss(device=device)

    # 创建两个随机图像张量，模拟输入
    x = torch.rand(1, 3, 224, 224).to(device)  # 图像 1
    y = torch.rand(1, 3, 224, 224).to(device)  # 图像 2

    loss = criterion(x, y)
    print(f'LPIPS Loss: {loss.item():.4f}')
