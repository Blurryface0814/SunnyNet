import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import se_block, cbam_block, eca_block


class SunnyNet(nn.Module):
    """
    Arguments:
        num_classes (int): number of output classes
        attention_type (string): attention type
    """

    def __init__(self, num_classes=4, attention_type=None):
        super(SunnyNet, self).__init__()
        self.attention_type = attention_type
        # b c h w = 10 1 32 400
        self.sunny1 = SunnyBlock(2, 32, self.attention_type)
        self.sunny2 = SunnyBlock(32, 64, self.attention_type)
        self.sunny3 = SunnyBlock(64, 96, self.attention_type)
        self.sunny4 = SunnyBlock(96, 96, self.attention_type)
        self.drop_layer = nn.Dropout()
        self.sunny5 = SunnyBlock(96, 64, self.attention_type)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, distance, reflectivity):
        # print("distance: '{}'".format(distance.shape))
        # print("reflectivity: '{}'".format(reflectivity.shape))

        # b c h w = 10 1 32 400 -> 10 2 32 400
        x = torch.cat([distance, reflectivity], 1)

        # b c h w = 10 2 32 400 -> 10 32 32 400
        x = self.sunny1(x)

        # b c h w = 10 32 32 400 -> 10 64 32 400
        x = self.sunny2(x)

        # b c h w = 10 64 32 400 -> 10 96 32 400
        x = self.sunny3(x)

        # b c h w = 10 96 32 400 -> 10 96 32 400
        x = self.sunny4(x)

        # b c h w = 10 96 32 400 -> 10 96 32 400
        x = self.drop_layer(x)

        # b c h w = 10 96 32 400 -> 10 64 32 400
        x = self.sunny5(x)

        # b c h w = 10 64 32 400 -> 10 4 32 400
        x = self.classifier(x)

        return x


class SunnyBlock(nn.Module):

    def __init__(self, in_channels, n, attention_type=None):
        super(SunnyBlock, self).__init__()

        self.branch1 = BasicConv2d(in_channels, n, kernel_size=(7, 3), padding=(2, 0), stride=(1, 1))
        self.branch2 = BasicConv2d(in_channels, n, kernel_size=3, stride=(1, 1))
        self.branch3 = BasicConv2d(in_channels, n, kernel_size=3, dilation=(2, 2), padding=1, stride=(1, 1))
        self.branch4 = BasicConv2d(in_channels, n, kernel_size=(3, 7), padding=(0, 2), stride=(1, 1))
        self.conv = BasicConv2d(n * 4, n, kernel_size=1, padding=1, stride=(1, 1))
        self.attention_type = attention_type

        # add attention_block
        if self.attention_type == 'cbam':
            self.attention = cbam_block(n * 4, ratio=8, kernel_size=7)
        elif self.attention_type == 'eca':
            self.attention = eca_block(n * 4, b=1, gamma=2)
        elif self.attention_type == 'senet':
            self.attention = se_block(n * 4, ratio=16)
        elif self.attention_type == 'original':
            pass

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = torch.cat([branch1, branch2, branch3, branch4], 1)

        # add attention_block
        if self.attention_type != 'original':
            output = self.attention(output)

        output = self.conv(output)
        return output


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__ == '__main__':
    num_classes, height, width = 4, 64, 512

    model = SunnyNet(num_classes)  # .to('cuda')
    inp = torch.randn(5, 1, height, width)  # .to('cuda')

    out = model(inp, inp)
    assert out.size() == torch.Size([5, num_classes, height, width])

    print('Pass size check.')
