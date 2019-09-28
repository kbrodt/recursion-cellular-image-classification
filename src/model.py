import math

from catalyst.contrib.modules import GlobalConcatAttnPool2d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DenseNet121(nn.Module):
    def __init__(self, num_classes=1000, num_channels=6, with_plates=False):
        super().__init__()
        
        self.with_plates = with_plates
        preloaded = torchvision.models.densenet121(pretrained=True)
        self.bn = nn.BatchNorm2d(num_channels)
        self.features = preloaded.features
        if num_channels != 3:
            self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        n_plates = 4 if with_plates else 0
        self.classifier = nn.Linear(1024 + n_plates, num_classes, bias=True)
        del preloaded
        
    def forward(self, x):
        if self.with_plates:
            x, p = x
        x = self.bn(x)
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        if self.with_plates:
            x = torch.cat([x, p], dim=-1)
        
        return self.classifier(x)

    
class DenseNet121All(nn.Module):
    def __init__(self, num_classes=1000, num_channels=6, with_plates=False):
        super().__init__()
        
        self.with_plates = with_plates
        preloaded = torchvision.models.densenet121(pretrained=True)
        self.bn = nn.BatchNorm2d(num_channels)
        self.features = preloaded.features
        if num_channels != 3:
            self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        n_plates = 4 if with_plates else 0
        self.classifier = nn.Linear(1024 + n_plates, num_classes, bias=True)
        del preloaded
        
    def forward(self, x):
        if self.with_plates:
            x, p = x
        x = self.bn(x)
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        if self.with_plates:
            x = torch.cat([x, p], dim=-1)
        
        return self.classifier(x)
    

class DenseNet121New(nn.Module):
    def __init__(self, num_classes=1000, num_channels=6, with_plates=False):
        super().__init__()
        
        self.with_plates = with_plates
        self.bn = nn.BatchNorm2d(num_channels)
        preloaded = torchvision.models.densenet121(pretrained=True)
        self.features = preloaded.features
        if num_channels != 3:
            self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        in_features = preloaded.classifier.in_features
        self.relu = nn.ReLU(inplace=True)
        self.pool = GlobalConcatAttnPool2d(in_features)
        n_plates = 4 if with_plates else 0
        self.pre_classifier = nn.Sequential(
            nn.BatchNorm1d(3*in_features + n_plates),
            nn.Linear(in_features=3*in_features + n_plates, out_features=1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        
        del preloaded
        
    def forward(self, x):
        if self.with_plates:
            x, p = x
        x = self.bn(x)
        x = self.features(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        if self.with_plates:
            x = torch.cat([x, p], dim=-1)
        
        x = self.pre_classifier(x)
        
        return self.classifier(x)
    

class DenseNet201(nn.Module):
    def __init__(self, num_classes=1000, num_channels=6, with_plates=False):
        super().__init__()
        
        self.with_plates = with_plates
        preloaded = torchvision.models.densenet201(pretrained=True)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        n_plates = 4 if with_plates else 0
        self.classifier = nn.Linear(1920 + n_plates, num_classes, bias=True)
        del preloaded
        
    def forward(self, x):
        if self.with_plates:
            x, p = x
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        
        if self.with_plates:
            out = torch.cat([out, p], dim=-1)
        
        return self.classifier(out)


class LargeMarginCosineLoss(nn.Module):
    def __init__(self, m=0.25, s=64):
        super().__init__()
        self.m = m
        self.s = s
        self.xent = nn.CrossEntropyLoss()

    def forward(self, input, target):
        input = input.clone()
        input[range(len(target)), target] -= self.m
        input *= self.s

        return self.xent(input, target)
    
    
class CosineBlock(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(CosineBlock, self).__init__()
        self.feature_size = feature_size
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.randn(self.num_classes, self.feature_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features):
        features_norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        features_normalized = torch.div(features, features_norm)

        weight_norm = torch.norm(self.weight, p=2, dim=-1, keepdim=True)
        weight_normalized = torch.div(self.weight, weight_norm)
        weight_normalized.transpose_(0, 1)

        logits = torch.matmul(features_normalized, weight_normalized)

        return logits


class LargeMarginCosineNet(nn.Module):
    def __init__(self, config, embedding_net, input_features, num_classes):
        super(LargeMarginCosineNet, self).__init__()
        self.config = config
        self.probability = 0.0
        self.embedding_net = embedding_net
        self.input_features = input_features
        self.num_classes = num_classes

        self.mixed_precision_flag = False

        self.last_linear = nn.Sequential(
            CosineBlock(self.input_features, self.num_classes)
        )

        self.io_tuple = False
        self.softmax = False

    def parameters_with_lrs(self, lrs=(0.1, 0.3, 1.0)):
        parameters = self.embedding_net.parameters_with_lrs(lrs)

        for module in self.last_linear.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                parameters.append(
                    {'params': module.parameters(), 'lr_factor': lrs[-1], 'enable_weight_decay': False}
                )
            elif len(list(module.children())) == 0:
                parameters.append(
                    {'params': module.parameters(), 'lr_factor': lrs[-1], 'enable_weight_decay': True}
                )

        return parameters

    def output_result(self, output):
        if self.io_tuple:
            return output,
        else:
            return output

    def input_x(self, x):
        if self.io_tuple:
            x, = x
        return x

    def forward(self, x):
        return self._normal_forward(x)

    def _normal_forward(self, x):
        x = self.input_x(x)
        output = self.embedding_net(x)

        if self.mixed_precision_flag:
            output = output.float()

        output = self.last_linear(output)

        if self.softmax:
            output = F.softmax(output, dim=1)

        return self.output_result(output)

    def get_features(self, x):
        x = self.input_x(x)
        output = self.embedding_net(x)

        if self.mixed_precision_flag:
            output = output.float()

#         output = self.last_linear[:-1](output)

        return self.output_result(output)

    def get_features_and_classes(self, x):
        x = self.input_x(x)
        embedding = self.embedding_net(x)

        if self.mixed_precision_flag:
            embedding = embedding.float()

        classes = self.last_linear(embedding)

        return embedding, classes

    def to_mixed_precision(self):
        self.mixed_precision_flag = True

        for module in self.embedding_net.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.half()


class SeResNext(nn.Module):
    def __init__(self, config, model_name, num_channels, pretrained=False, dropout=0.0):
        super(SeResNext, self).__init__()
        self.config = config

        if model_name == 'seresnext5032x4d':
            self.model_name = 'se_resnext50_32x4d'
        elif model_name == 'seresnext10132x4d':
            self.model_name = 'se_resnext101_32x4d'
        elif model_name == 'seresnet50':
            self.model_name = 'se_resnet50'
        elif model_name == 'seresnet101':
            self.model_name = 'se_resnet101'
        else:
            self.model_name = model_name

        self.pretrained = pretrained
        preloaded = torchvision.models.densenet121(pretrained=True)
        self.backbone = preloaded.features
        if num_channels != 3:
            self.backbone.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        del preloaded

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        return x


class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):  # pylint: disable=protected-access
    """
    Cosine annealing with restarts.
    This is decribed in the paper https://arxiv.org/abs/1608.03983.
    Parameters
    ----------
    optimizer : ``torch.optim.Optimizer``
    t_max : ``int``
        The maximum number of iterations within the first cycle.
    eta_min : ``float``, optional (default=0)
        The minimum learning rate.
    last_epoch : ``int``, optional (default=-1)
        The index of the last epoch. This is used when restarting.
    factor : ``float``, optional (default=1)
        The factor by which the cycle length (``T_max``) increases after each restart.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        assert t_max > 0
        assert eta_min >= 0
        if t_max == 1 and factor == 1:
            print("Cosine annealing scheduler will have no effect on the learning "
                           "rate since T_max = 1 and factor = 1.")
        self.t_max = t_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.
        self._updated_cycle_len = t_max
        self._initialized = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time ``self.get_lr()`` was called,
        # since ``torch.optim.lr_scheduler._LRScheduler`` will call ``self.get_lr()``
        # when first initialized, but the learning rate should remain unchanged
        # for the first epoch.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
                self.eta_min + ((lr - self.eta_min) / 2) * (
                        np.cos(
                                np.pi *
                                (self._cycle_counter % self._updated_cycle_len) /
                                self._updated_cycle_len
                        ) + 1
                )
                for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.t_max)
            self._last_restart = step

        return lrs
    

def get_model(name, num_classes, num_channels=6, with_plates=False):
    if name == 'densenet121':
        return DenseNet121(num_classes=num_classes, num_channels=num_channels, with_plates=with_plates)
    if name == 'densenet121new':
        return DenseNet121New(num_classes=num_classes, num_channels=num_channels, with_plates=with_plates)
    elif name == 'densenet201':
        return DenseNet201(num_classes=num_classes, num_channels=num_channels, with_plates=with_plates)
    elif name == 'seresnet50':
        print('LargeCosBlock')
        backbone = SeResNext(
            None,
            model_name=name,
            num_channels=num_channels,
            pretrained=True,
            dropout=0.0
        )
        
        return LargeMarginCosineNet(None, backbone, input_features=1024, num_classes=num_classes)
    else:
        raise(f'No such model: {name}')
