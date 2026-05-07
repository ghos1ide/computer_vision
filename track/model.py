"""U-Net segmentation model with optional SE attention blocks."""

import jittor as jt
from jittor import nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv(in_channels, out_channels, 3, stride=1, padding=1, bias=False)
        # 针对 batch_size=4 这种小批次，GroupNorm 比 BatchNorm 更稳定
        self.bn = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv(channels, hidden, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv(hidden, channels, 1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        w = self.pool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_se: bool) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)
        self.se = SEBlock(out_channels) if use_se else None
        # 避免梯度消失，加入残差连接（Residual Connection）
        self.proj = nn.Conv(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else None

    def execute(self, x):
        res = x if self.proj is None else self.proj(x)
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        return out + res


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_se: bool) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, use_se=use_se)

    def execute(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_se: bool) -> None:
        super().__init__()
        self.up = nn.ConvTranspose(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels, use_se=use_se)

    def execute(self, x, skip):
        x = self.up(x)
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = nn.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode="bilinear", align_corners=False)
        x = jt.contrib.concat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """UNet with Pre-trained ResNet-18 Encoder (Plan B)"""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        base_channels: int = 32, # Kept for API compatibility
        use_se: bool = True,
    ) -> None:
        super().__init__()
        import jittor.models as jm
        
        # 加载官方在 ImageNet 上预训练好的 ResNet-18
        resnet = jm.resnet18(pretrained=True)
        
        # 编码器 (Encoder / Backbone): 使用 ResNet 的各个 Stage
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 输出通道数: 64,  分辨率: H/4
        self.layer2 = resnet.layer2  # 输出通道数: 128, 分辨率: H/8
        self.layer3 = resnet.layer3  # 输出通道数: 256, 分辨率: H/16
        self.layer4 = resnet.layer4  # 输出通道数: 512, 分辨率: H/32

        # 解码器 (Decoder): 与 ResNet 特征图对应的上采样层
        self.up1 = UpBlock(512, 256, use_se=use_se) # (H/32 -> H/16), 拼接 layer3
        self.up2 = UpBlock(256, 128, use_se=use_se) # (H/16 -> H/8) , 拼接 layer2
        self.up3 = UpBlock(128, 64, use_se=use_se)  # (H/8 -> H/4)  , 拼接 layer1
        self.up4 = UpBlock(64, 64, use_se=use_se)   # (H/4 -> H/2)  , 拼接 conv1 后的特征

        # 最后一层将 H/2 还原到原始尺寸 H
        self.up5_trans = nn.ConvTranspose(64, base_channels, kernel_size=2, stride=2)
        self.up5_conv = ConvBlock(base_channels, base_channels, use_se=use_se)

        # 最终映射到掩码类别数
        self.head = nn.Conv(base_channels, num_classes, 1, stride=1, padding=0)

    def execute(self, x):
        # --- Encoder ---
        # x: [B, 3, H, W]
        s1 = self.relu(self.bn1(self.conv1(x)))      # [B, 64,  H/2, W/2]
        s2 = self.layer1(self.maxpool(s1))           # [B, 64,  H/4, W/4]
        s3 = self.layer2(s2)                         # [B, 128, H/8, W/8]
        s4 = self.layer3(s3)                         # [B, 256, H/16, W/16]
        b  = self.layer4(s4)                         # [B, 512, H/32, W/32] 瓶颈层

        # --- Decoder ---
        u4 = self.up1(b, s4)                         # [B, 256, H/16, W/16]
        u3 = self.up2(u4, s3)                        # [B, 128, H/8, W/8]
        u2 = self.up3(u3, s2)                        # [B, 64,  H/4, W/4]
        u1 = self.up4(u2, s1)                        # [B, 64,  H/2, W/2]

        # --- Final Upsampling ---
        out = self.up5_trans(u1)                     # [B, base_channels, H, W]
        out = self.up5_conv(out)
        out = self.head(out)
        
        return out
