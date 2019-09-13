# from https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/blob/master/selim_sef

import os

import torch
from torch import nn
from torch.utils import model_zoo

from src.models.resnet import resnet34

encoder_params = {
    'resnet34':
        {
            'filters': [64, 64, 128, 256, 512],
            'decoder_filters': [64, 128, 256, 256],
            'last_upsample': 64,
            'init_op': resnet34,
            'url': None,
        }
}


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()[self.first_layer_params_name +
                               '.weight'][:, :3, ...] = pretrained_dict[self.first_layer_params_name + '.weight'].data
            skip_layers = [
                self.first_layer_params_name,
                self.first_layer_params_name + '.weight',
            ]
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if not any(k.startswith(s) for s in skip_layers)
            }
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_name(self):
        return 'conv1'


class EncoderDecoder(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34'):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        self.filters = encoder_params[encoder_name]['filters']
        self.decoder_filters = encoder_params[encoder_name].get('decoder_filters', self.filters[:-1])
        self.last_upsample_filters = encoder_params[encoder_name].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.bottlenecks = nn.ModuleList(
            [
                self.bottleneck_type(self.filters[-i - 2] + f, f)
                for i, f in enumerate(reversed(self.decoder_filters[:]))
            ]
        )

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])

        if self.first_layer_stride_two:
            self.last_upsample = self.decoder_block(
                self.decoder_filters[0],
                self.last_upsample_filters,
                self.last_upsample_filters,
            )

        self.final = self.make_final_classifier(
            self.last_upsample_filters if self.first_layer_stride_two else self.decoder_filters[0],
            num_classes,
        )
        self._initialize_weights()

        encoder = encoder_params[encoder_name]['init_op'](pretrained=True)
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
        if encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'], num_channels != 3)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # x, angles = x
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        last_dec_out = enc_results[-1]
        # size = last_dec_out.size(2)
        # last_dec_out = torch.cat([last_dec_out, F.upsample(angles, size=(size, size), mode="nearest")], dim=1)
        x = last_dec_out
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = -(idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        f = self.final(x)

        return f

    def get_decoder(self, layer):
        in_channels = (
            self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[layer + 1]
        )
        return self.decoder_block(
            in_channels,
            self.decoder_filters[layer],
            self.decoder_filters[max(layer, 0)],
        )

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(nn.Conv2d(in_filters, num_classes, 1, padding=0))

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _get_layers_params([self.encoder_stages[0]])

    @property
    def layers_except_first_params(self):
        layers = get_slice(self.encoder_stages, 1, -1) + [
            self.bottlenecks,
            self.decoder_stages,
            self.final,
        ]
        return _get_layers_params(layers)


def _get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])


def get_slice(features, start, end):
    if end == -1:
        end = len(features)
    return [features[i] for i in range(start, end)]


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class Resnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 4, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        elif layer == 1:
            return nn.Sequential(encoder.maxpool, encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class ResnetSuperVision(Resnet):
    def __init__(self, seg_classes, backbone_arch):
        super().__init__(seg_classes, backbone_arch=backbone_arch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        last_dec_out = enc_results[-1]
        x = last_dec_out

        x_cls = self.avgpool(x)
        x_cls = x_cls.view(x_cls.size(0), -1)
        x_cls = self.fc(x_cls).view(x_cls.size(0))

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = -(idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        f = self.final(x)

        return f, x_cls


if __name__ == '__main__':
    d = ResnetSuperVision(1, backbone_arch='resnet34')
    d.eval()
    import numpy as np

    with torch.no_grad():
        images = torch.from_numpy(np.zeros((4, 3, 256, 256), dtype='float32'))
        p1, p2 = d(images)
        print(p1.shape)
        print(p2.shape)

    print(d)
