import torch
from torch import nn
from torch.nn import functional as F

# import extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):    # 两倍上采样
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    """
    不同卷积核卷积之后cat,然后3次上采样
    INPUT:[2, 64, 20, 10]
    OUTPUT:[2, 64, 160, 80]
    """
    def __init__(self, in_channels=64, out_channels=64, sizes=(1, 2, 3, 6)):
        super().__init__()

        self.psp = PSPModule(in_channels, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, out_channels)
        self.up_3 = PSPUpsample(out_channels, out_channels)
        self.drop_2 = nn.Dropout2d(p=0.15)

    def forward(self, x):

        p = self.psp(x)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        return p



if __name__=="__main__":

    # net = PSPModule(64, 64, (1, 2, 3, 6))
    # input_data = torch.randn(2, 64, 20, 10)
    # out = net(input_data)
    # up_1 = PSPUpsample(64, 64)
    # out_up = up_1(out)

    # out_module=PSPNet()
    # out_net=out_module(input_data)

    # print(out.shape)
    # print(out_up.shape)
    # print(out_net.shape)

    network = PSPNet()
    test_input = torch.Tensor(1, 64, 640, 640)
    output = network(test_input)