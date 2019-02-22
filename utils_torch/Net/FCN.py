import math

import torch
import torch.nn as nn

__all__ = ['FCN']


class FCN(nn.Module):
    """
    input: 1, in_channels, 640, 640
    output: 1, num_labels, 640, 640
    """
    def __init__(self, in_channels=10, num_labels=4):
        super(FCN, self).__init__()
        self.channel_dict = {
            'conv_seg_1': [24, 24, 48, 48],
            'conv_seg_2': [64, 64, 64],
            'conv_seg_3': [96, 96, 96],
            'conv_seg_4': [128, 128, 128],
            'conv_seg_5': [192, 192],
            'dconv_seg_1': [192, 128],
            'dconv_seg_2': [128, 96],
            'dconv_seg_3': [96, 64],
            'dconv_seg_4': [64, 48],
            'dconv_seg_5': [48, num_labels],
        }
        self.conv_seg_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.channel_dict['conv_seg_1'][0],
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_1'][0], out_channels=self.channel_dict['conv_seg_1'][1],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_1'][1], out_channels=self.channel_dict['conv_seg_1'][2],
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_1'][2], out_channels=self.channel_dict['conv_seg_1'][3],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_seg_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_1'][3], out_channels=self.channel_dict['conv_seg_2'][0],
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_2'][0], out_channels=self.channel_dict['conv_seg_2'][1],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_2'][1], out_channels=self.channel_dict['conv_seg_2'][2],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_seg_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_2'][2], out_channels=self.channel_dict['conv_seg_3'][0],
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_3'][0], out_channels=self.channel_dict['conv_seg_3'][1],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_3'][1], out_channels=self.channel_dict['conv_seg_3'][2],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_seg_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_3'][2], out_channels=self.channel_dict['conv_seg_4'][0],
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_4'][0], out_channels=self.channel_dict['conv_seg_4'][1],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_4'][1], out_channels=self.channel_dict['conv_seg_4'][2],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_seg_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_4'][2], out_channels=self.channel_dict['conv_seg_5'][0],
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_5'][0], out_channels=self.channel_dict['conv_seg_5'][1],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dconv_seg_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_5'][1], out_channels=self.channel_dict['dconv_seg_1'][0],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_dict['dconv_seg_1'][0], out_channels=self.channel_dict['dconv_seg_1'][1],
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # cat
        self.dconv_seg_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_4'][-1]+self.channel_dict['dconv_seg_1'][1],
                      out_channels=self.channel_dict['dconv_seg_2'][0],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_dict['dconv_seg_2'][0], out_channels=self.channel_dict['dconv_seg_2'][1],
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # cat
        self.dconv_seg_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_3'][-1]+self.channel_dict['dconv_seg_2'][1],
                      out_channels=self.channel_dict['dconv_seg_3'][0],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_dict['dconv_seg_3'][0], out_channels=self.channel_dict['dconv_seg_3'][1],
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # cat
        self.dconv_seg_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_2'][-1]+self.channel_dict['dconv_seg_3'][1],
                      out_channels=self.channel_dict['dconv_seg_4'][0],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_dict['dconv_seg_4'][0], out_channels=self.channel_dict['dconv_seg_4'][1],
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # cat
        self.dconv_seg_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dict['conv_seg_1'][-1]+self.channel_dict['dconv_seg_4'][1],
                      out_channels=self.channel_dict['dconv_seg_5'][0],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_dict['dconv_seg_5'][0], out_channels=self.channel_dict['dconv_seg_5'][1],
                               kernel_size=4, stride=2, padding=1),
            # nn.ReLU(inplace=True)
        )
        self._init_weight()
        self.dump_patches = True

    # initialize weights
    def _init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                n = layer.kernel_size[0] * \
                    layer.kernel_size[1] * layer.out_channels
                layer.weight.data.normal_(0, math.sqrt(2. / n))
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    # forward propagation
    def forward(self, x):
        shallow_feature_1 = self.conv_seg_1(x)
        shallow_feature_2 = self.conv_seg_2(shallow_feature_1)
        shallow_feature_3 = self.conv_seg_3(shallow_feature_2)
        shallow_feature_4 = self.conv_seg_4(shallow_feature_3)
        shallow_feature_5 = self.conv_seg_5(shallow_feature_4)
        output_feature = self.dconv_seg_1(shallow_feature_5)
        output_feature = torch.cat((shallow_feature_4, output_feature), 1)
        output_feature = self.dconv_seg_2(output_feature)
        output_feature = torch.cat((shallow_feature_3, output_feature), 1)
        output_feature = self.dconv_seg_3(output_feature)
        output_feature = torch.cat((shallow_feature_2, output_feature), 1)
        output_feature = self.dconv_seg_4(output_feature)
        output_feature = torch.cat((shallow_feature_1, output_feature), 1)
        output_feature = self.dconv_seg_5(output_feature)
        return output_feature



# check model
def main():
    network = FCN()
    test_input = torch.Tensor(1, 10, 640, 640)
    output = network(test_input)
    print(network)
    print(output.size())

if __name__ == '__main__':
    main()

