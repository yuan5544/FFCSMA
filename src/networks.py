import torch
import torch.nn as nn
from kpn.network import KernelConv
import kpn.utils as kpn_utils
import numpy as np
from src.moduels import FFCResnetBlock ,ESP_Attn
from src.ffc import FFCResnetBlock
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)






class InpaintGenerator(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.filter_type = config.FILTER_TYPE
        self.kernel_size = config.kernel_size

        self.encoder0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )



        blocks = []
        # for _ in range(residual_blocks):
        #     block = ResnetBlock(256, 2)
        #     blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)

        self.kpn_model = kpn_utils.create_generator()

        self.ESP_Attn = ESP_Attn(in_channels = 256 , in_width = 64, in_height = 64 , token_facter = 3 )
        self.FFCResnetBlock= FFCResnetBlock(dim = 256, padding_type = 'reflect', norm_layer = nn.BatchNorm2d)
        self.Linear1 = nn.Linear(256,64)
        self.Linear2 = nn.Linear(256,192)
        self.Linear3 = nn.Linear(64,192)
        self.convc = nn.Conv2d(4,256,1,1)
        self.convc1 = nn.Conv2d(4,256,1,1)
        self.convc2 = nn.Conv2d(3,4,1,1)
        if init_weights:
            self.init_weights()

    def forward(self, x,mask):
        inputs = x.clone()

        x = self.encoder0(x) # 64*256*256
        x = self.encoder1(x) # 128*128*128

        kernels, kernels_img = self.kpn_model(inputs, x)

        x = self.encoder2(x) # 256*64*64

        x1 = x.reshape(1,64,64,256)
        # x1 = self.Linear1(x)
        x2 = self.Linear2(x1)
        x3 = self.Linear1(x1)
        # x = x.reshape(1,256,70,70)
        # x1 = x1.reshape(1,64,70,70)
        x2 = x2.reshape(1,192,64,64)
        x3 = x3.reshape(1,64,64,64)
        x4 = (x2,x3)
        x4 = self.FFCResnetBlock(x4)
        x1 = x4.reshape(1,64,64,256)
        x2 = self.Linear2(x1)
        x3 = self.Linear1(x1)
        x2 = x2.reshape(1,192,64,64)
        x3 = x3.reshape(1,64,64,64)
        x4 = (x2,x3)
        x1 = self.FFCResnetBlock(x4)

        fin = x.clone()
        h, w = fin.shape[2], fin.shape[3]
        subcontext = F.interpolate(inputs, (h, w), mode='bilinear', align_corners = False)
        subcontext = self.convc(subcontext)
        mask = self.convc2(mask)
        submask = F.interpolate(mask, (h, w), mode='nearest')
        submask = self.convc1(submask)
        espa_in = fin * submask + subcontext * (1. - submask)
        espa_out = self.ESP_Attn(espa_in)
        # espa_out = espa_out* submask + subcontext * (1. - submask)

        x = x1 + espa_out
        x = self.kernel_pred(x, kernels, white_level=1.0, rate=1)

        x = self.middle(x) # 256*64*64

        x = self.decoder(x) # 3*256*256

        x = self.kernel_pred(x, kernels_img, white_level=1.0, rate=1)

        x = (torch.tanh(x) + 1) / 2

        return x

    def save_feature(self, x, name):
        x = x.cpu().numpy()
        np.save('./result/{}'.format(name), x)


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


# class ResnetBlock(nn.Module):
#     def __init__(self, dim, dilation=1, use_spectral_norm=False):
#         super(ResnetBlock, self).__init__()
#         self.conv_block = nn.Sequential(
#             nn.ReflectionPad2d(dilation),
#             spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
#             nn.InstanceNorm2d(dim, track_running_stats=False),
#             nn.ReLU(True),

#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
#             nn.InstanceNorm2d(dim, track_running_stats=False),
#         )

#     def forward(self, x):
#         out = x + self.conv_block(x)

#         return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


if __name__ == '__main__':
    x = torch.rand(2, 3, 512, 512).cuda()
    model = Discriminator(3).cuda()
    out, _ = model(x)
    print(out.shape)