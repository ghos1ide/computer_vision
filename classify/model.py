import jittor as jt
from jittor import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.3, use_batchnorm=True, use_dropout=True):
        super(ResidualBlock, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        conv_bias = not use_batchnorm
        self.conv1 = nn.Conv(in_channels, out_channels, 3, stride, 1, bias=conv_bias)
        self.bn1 = nn.BatchNorm(out_channels) if use_batchnorm else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob) if use_dropout else None
        self.conv2 = nn.Conv(out_channels, out_channels, 3, 1, 1, bias=conv_bias)
        self.bn2 = nn.BatchNorm(out_channels) if use_batchnorm else None
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample_layers = [nn.Conv(in_channels, out_channels, 1, stride, 0, bias=conv_bias)]
            if use_batchnorm:
                downsample_layers.append(nn.BatchNorm(out_channels))
            self.downsample = nn.Sequential(*downsample_layers)

    def execute(self, x):
        identity = x
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0.3, use_batchnorm=True, use_dropout=True):
        super(SimpleResNet, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        conv_bias = not use_batchnorm
        self.conv1 = nn.Conv(3, 64, 3, 1, 1, bias=conv_bias)
        self.bn1 = nn.BatchNorm(64) if use_batchnorm else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob) if use_dropout else None
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout_prob=dropout_prob, use_batchnorm=use_batchnorm, use_dropout=use_dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout_prob=dropout_prob, use_batchnorm=use_batchnorm, use_dropout=use_dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout_prob=dropout_prob, use_batchnorm=use_batchnorm, use_dropout=use_dropout)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout_prob=dropout_prob, use_batchnorm=use_batchnorm, use_dropout=use_dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout_prob, use_batchnorm, use_dropout):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_prob, use_batchnorm, use_dropout))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, dropout_prob, use_batchnorm, use_dropout))
        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = jt.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        return x
