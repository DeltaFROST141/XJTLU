from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Residual block of ResNet"""

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):

        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(
                inchannel,
                outchannel,
                3,
                stride,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                outchannel,
                outchannel,
                3,
                1,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(outchannel),
        )
        self.right = shortcut

    def forward(self, x):
        """Forward function"""
        out = self.left(x)
        if self.right is None:
            residual = x
        else:
            residual = self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):
    """ResNet34"""

    def __init__(self):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = self.make_layer(64, 128, 4)
        self.layer2 = self.make_layer(128, 256, 4, stride=2)
        self.layer3 = self.make_layer(256, 256, 6, stride=2)
        self.layer4 = self.make_layer(256, 512, 3, stride=2)
        self.fc = nn.Linear(512, 1)
        self.activation = nn.Sigmoid()

    def make_layer(self, inchannel, outchannel, block_num, stride=1):
        """Make layer"""
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function"""
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.activation(x)
