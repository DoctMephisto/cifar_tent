import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_cfg='VGG11'):
        super(VGG, self).__init__()
        self.features = self._make_layer(cfg[vgg_cfg])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        return out

    def _make_layer(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_features=x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def test():
    net = VGG()
    print(net)
    x = torch.randn((10, 3, 32, 32))
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()
