import torch

# The structure of a classic Residual Block, used in ResNet18 and ResNet34
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.dropout = dropout
        self.dropout1 = torch.nn.Dropout(p=self.dropout)
        self.dropout2 = torch.nn.Dropout(p=self.dropout)
        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
                        torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        torch.nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = torch.nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.dropout1(self.conv1(x))
        out = self.dropout2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# The structure of a bottleneck Residual Block, used in ResNet50 and ResNet101
class BottleneckBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels * 4, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = torch.nn.BatchNorm2d(out_channels * 4)

        self.relu = torch.nn.ReLU()

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != 4 * out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, 4 * out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(4 * out_channels)
            )
        
    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += self.shortcut(residual)
        x = self.relu(x)

        return x

class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes = 10, dropout=0.0):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        torch.nn.BatchNorm2d(64),
                        torch.nn.ReLU())
        self.maxpool = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1, dropout=dropout)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2, dropout=dropout)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2, dropout=dropout)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2, dropout=dropout)
        self.avgpool = torch.nn.AvgPool2d(7, stride=1)
        self.fc = torch.nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1, dropout=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout=dropout))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout=dropout))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        x = self.fc(feat)

        return x

class ResNet50(torch.nn.Module):
    def __init__(self, block, layers, num_classes = 10, dropout = 0):
        super(ResNet, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(64, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(256,128, layers[1], stride = 2)
        self.layer2 = self._make_layer(512, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(1024, 512, layers[3], stride = 2)
        self.avgpool = torch.nn.AvgPool2d(7, stride=1)
        self.fc = torch.nn.Linear(512 * 4, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for stride in strides:
            layers.append(ResidualBlock(out_channels * 4, out_channels, 1))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # torch.Size([N, 3, 224, 224])
        x = self.conv1(x)
        # torch.Size([N, 64, 112, 112])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        # torch.Size([2, 64, 56, 56])

        x = self.layer0(x)
        # torch.Size([2, 256, 56, 56])
        x = self.dropout(x)
        x = self.layer1(x)
        # torch.Size([2, 512, 28, 28])
        x = self.dropout(x)
        x = self.layer2(x)
        # torch.Size([2, 1024, 14, 14])
        x = self.dropout(x)
        x = self.layer3(x)
        # torch.Size([2, 2048, 7, 7])
        x = self.dropout(x)


        x = self.avgpool(x)
        # torch.Size([2, 2048, 1, 1])
        feat = torch.flatten(x, 1)
        # torch.Size([2, 2048])
        x = self.fc(feat)
        # torch.Size([2, 10])
        return x
