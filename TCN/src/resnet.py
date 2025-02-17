import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Bottleneck block for ResNet architectures with 50 layers or more
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, norm_type='batch'):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = self._get_norm_layer(out_channels, norm_type)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = self._get_norm_layer(out_channels, norm_type)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = self._get_norm_layer(out_channels * self.expansion, norm_type)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def _get_norm_layer(self, num_features, norm_type):
        if norm_type == 'batch':
            return nn.BatchNorm2d(num_features)
        elif norm_type == 'group':
            num_groups = 32
            return nn.GroupNorm(num_groups, num_features)
        else:
            raise ValueError("Invalid normalization type. Use 'batch' or 'group'.")

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x


# Define a simpler Block for ResNet-18 and ResNet-34
class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, norm_type='batch'):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = self._get_norm_layer(out_channels, norm_type)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = self._get_norm_layer(out_channels, norm_type)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def _get_norm_layer(self, num_features, norm_type):
        if norm_type == 'batch':
            return nn.BatchNorm2d(num_features)
        elif norm_type == 'group':
            num_groups = 32
            return nn.GroupNorm(num_groups, num_features)
        else:
            raise ValueError("Invalid normalization type. Use 'batch' or 'group'.")

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


# Define the main ResNet class
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3, norm_type='batch', embedding_size=None):
        super(ResNet, self).__init__()
        self.embedding_size = embedding_size or num_classes
        print("Embedding Size: ", self.embedding_size)
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = self._get_norm_layer(64, norm_type)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64, norm_type=norm_type)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2, norm_type=norm_type)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2, norm_type=norm_type)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2, norm_type=norm_type)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, self.embedding_size)

    def _get_norm_layer(self, num_features, norm_type):
        if norm_type == 'batch':
            return nn.BatchNorm2d(num_features)
        elif norm_type == 'group':
            num_groups = 32
            return nn.GroupNorm(num_groups, num_features)
        else:
            raise ValueError("Invalid normalization type. Use 'batch' or 'group'.")

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1, norm_type='batch'):
        i_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            i_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                self._get_norm_layer(planes * ResBlock.expansion, norm_type)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=i_downsample, stride=stride, norm_type=norm_type))
        self.in_channels = planes * ResBlock.expansion

        for _ in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes, norm_type=norm_type))

        return nn.Sequential(*layers)


# ResNet model instantiations
def resnet18(num_classes, norm_type='batch', num_channels=3, embedding_size=None, **kwargs):
    embeddings = embedding_size or num_classes
    return ResNet(Block, [2, 2, 2, 2], num_classes=num_classes, num_channels=num_channels, norm_type=norm_type, embedding_size=embeddings, **kwargs)

def resnet34(num_classes, norm_type='batch', num_channels=3, embedding_size=None, **kwargs):
    embeddings = embedding_size or num_classes
    return ResNet(Block, [3, 4, 6, 3], num_classes=num_classes, num_channels=num_channels, norm_type=norm_type, embedding_size=embeddings, **kwargs)

def resnet50(num_classes, norm_type='batch', num_channels=3, embedding_size=None, **kwargs):
    embeddings = embedding_size or num_classes
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, num_channels=num_channels, norm_type=norm_type, embedding_size=embeddings, **kwargs)

def resnet101(num_classes, norm_type='batch', num_channels=3, embedding_size=None, **kwargs):
    embeddings = embedding_size or num_classes
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, num_channels=num_channels, norm_type=norm_type, embedding_size=embeddings, **kwargs)

def resnet152(num_classes, norm_type='batch', num_channels=3, embedding_size=None, **kwargs):
    embeddings = embedding_size or num_classes
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, num_channels=num_channels, norm_type=norm_type, embedding_size=embeddings, **kwargs)
