import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# DenseNet-B
class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP
        self.fc_in = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.act1 = nn.ReLU()
        self.fc_out = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sigmoid(self.fc_out(self.act1(self.fc_in(self.avg_pool(x)))))
        max_out = self.sigmoid(self.fc_out(self.act1(self.fc_in(self.max_pool(x)))))

        out = avg_out + max_out

        return out


#Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        self.padding = 3 if kernel_size == 7 else 1

        self.cov1 = nn.Conv2d(2, 1, kernel_size, padding=self.padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # concat
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.cov1(x)
        return self.sigmoid(x)


class CMBABlock(nn.Module):
    def __init__(self, in_channel, ratio=16, kernel_size=7):
        super(CMBABlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channel=in_channel, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out + res


# DenseNet-B
class _Bottleneck(nn.Module):

    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool, use_vision_attention: bool = False):
        super(_Bottleneck, self).__init__()
        self.use_vision_attention = use_vision_attention
        interChannels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(n_channels,
                               interChannels,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        if self.use_vision_attention:
            self.vision_attention1 = CMBABlock(interChannels)
        self.conv2 = nn.Conv2d(interChannels,
                               growth_rate,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        if self.use_vision_attention:
            self.vision_attention2 = CMBABlock(growth_rate)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_vision_attention:
            out = self.vision_attention1(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_vision_attention:
            out = self.vision_attention2(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# single layer
class _SingleLayer(nn.Module):

    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool, use_vision_attention: bool = False):
        super(_SingleLayer, self).__init__()
        self.use_vision_attention = use_vision_attention
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels,
                               growth_rate,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        if use_vision_attention:
            self.vision_attention = CMBABlock(growth_rate)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_vision_attention:
            out = self.vision_attention(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class _Transition(nn.Module):

    def __init__(self, n_channels: int, n_out_channels: int,
                 use_dropout: bool, use_vision_attention: bool = False):
        super(_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.use_vision_attention = use_vision_attention
        self.conv1 = nn.Conv2d(n_channels,
                               n_out_channels,
                               kernel_size=1,
                               bias=False)
        if use_vision_attention:
            self.vision_attention = CMBABlock(n_out_channels)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_vision_attention:
            out = self.vision_attention(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class DenseNet(nn.Module):

    def __init__(
            self,
            growth_rate: int,
            num_layers: int,
            reduction: float = 0.5,
            bottleneck: bool = True,
            use_dropout: bool = True,
            use_vision_attention: bool = False,
    ):
        super(DenseNet, self).__init__()
        n_dense_blocks = num_layers
        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(1,
                               n_channels,
                               kernel_size=7,
                               padding=3,
                               stride=2,
                               bias=False)
        self.norm1 = nn.BatchNorm2d(n_channels)
        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks,
                                       bottleneck, use_dropout, use_vision_attention)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout, use_vision_attention)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks,
                                       bottleneck, use_dropout, use_vision_attention)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout, use_vision_attention)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks,
                                       bottleneck, use_dropout, use_vision_attention)

        self.out_channels = n_channels + n_dense_blocks * growth_rate
        self.post_norm = nn.BatchNorm2d(self.out_channels)

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck,
                    use_dropout, use_vision_attention):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate,
                                          use_dropout, use_vision_attention))
            else:
                layers.append(
                    _SingleLayer(n_channels, growth_rate, use_dropout, use_vision_attention))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, x_mask):
        out = self.conv1(x)
        out = self.norm1(out)
        # out_mask = x_mask[:, 0::2, 0::2]
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        # print("Before Dense1 feature shape: ", out.shape)
        # out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense1(out)
        out = self.trans1(out)
        # out_mask = out_mask[:, 0::2, 0::2]
        # print("Before Dense2 feature shape: ", out.shape)
        out = self.dense2(out)
        out = self.trans2(out)
        # print("Before Dense3 feature shape: ", out.shape)
        out_mask = x_mask[:, 0::16, 0::16]
        out = self.dense3(out)
        out = self.post_norm(out)
        # print("After Dense3 feature shape: ", out.shape)
        return out, out_mask
    
    
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super(ProjectionHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Adjust the in_features of the first fully connected layer
        self.fc1 = nn.Linear(in_dim, 1024)  # Changed from 684 to in_dim
        self.bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, out_dim)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor (batch_size, in_dim)
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
class Model(nn.Module):
  def __init__(self, growth_rate: int, num_layers: int, reduction: float = 0.5, bottleneck: bool = True, use_dropout: bool = True,use_vision_attention: bool = False):
    super(Model, self).__init__()

    self.backbone = DenseNet(growth_rate, num_layers, reduction, bottleneck, use_dropout)
    self.projection_head = ProjectionHead(self.backbone.out_channels, out_dim=512)

  def forward(self, x, x_mask):
    x, x_mask = self.backbone(x, x_mask)
    x_proj = self.projection_head(x)
    return x, x_proj, x_mask