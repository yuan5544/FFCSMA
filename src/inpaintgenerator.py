import torch
import torch.nn as nn
import torch.nn.functional as F

from basenet import BaseNetwork
from kpn import KPN
from ffc import FFCResnetBlock
from espa import ESP_Attn
from kernel_net import KernelConv


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintGenerator, self).__init__()

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
        self.kpn = KPN()

        self.Linear1 = nn.Linear(256, 64)
        self.Linear2 = nn.Linear(256, 192)

        self.FFCResnetBlock = FFCResnetBlock(dim=256, padding_type='reflect', norm_layer=nn.BatchNorm2d)

        self.convc = nn.Conv2d(4, 256, 1, 1)
        self.convm3_4 = nn.Conv2d(3, 4, 1, 1)
        self.convm = nn.Conv2d(4, 256, 1, 1)

        self.ESP_Attn = ESP_Attn(in_channels=256, in_width=64, in_height=64, token_facter=3)

        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)

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

        if init_weights:
            self.init_weights()

    def forward(self, x, mask):
        inputs = x.clone()

        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        # print(x0.shape, x1.shape, x2.shape) # torch.Size([2, 64, 256, 256]) torch.Size([2, 128, 128, 128]) torch.Size([2, 256, 64, 64])

        k1, k2 = self.kpn(inputs, x1)
        # print(k1.shape, k2.shape) # torch.Size([2, 2304, 64, 64]) torch.Size([2, 27, 256, 256])

        x2 = x2.permute(0, 2, 3, 1)
        x2_1 = self.Linear2(x2)
        x2_2 = self.Linear1(x2)
        x2_1 = x2_1.permute(0, 3, 1, 2)
        x2_2 = x2_2.permute(0, 3, 1, 2)
        x2_ = (x2_1, x2_2)
        # print(x2_1.shape, x2_2.shape) # torch.Size([2, 192, 64, 64]) torch.Size([2, 64, 64, 64])
        x_ffc = self.FFCResnetBlock(x2_)
        # print(x_ffc.shape) # torch.Size([2, 256, 64, 64])

        x_ffc = x_ffc.permute(0, 2, 3, 1)
        x2_1 = self.Linear2(x_ffc)
        x2_2 = self.Linear1(x_ffc)
        x2_1 = x2_1.permute(0, 3, 1, 2)
        x2_2 = x2_2.permute(0, 3, 1, 2)
        x2_ = (x2_1, x2_2)
        # print(x2_1.shape, x2_2.shape) # torch.Size([2, 192, 64, 64]) torch.Size([2, 64, 64, 64])
        x_ffc = self.FFCResnetBlock(x2_)

        x2 = x2.permute(0, 3, 1, 2)
        fin = x2.clone()
        h, w = fin.shape[2], fin.shape[3]
        # print(h, w) # 64 64
        subcontext = F.interpolate(inputs, (h, w), mode='bilinear', align_corners=False)
        # print(subcontext.shape) # torch.Size([2, 4, 64, 64])
        subcontext = self.convc(subcontext)
        # print(subcontext.shape) # torch.Size([2, 256, 64, 64])

        submask = self.convm3_4(mask)
        # print(submask.shape) # torch.Size([2, 4, 256, 256])
        submask = F.interpolate(submask, (h, w), mode='nearest')
        # print(submask.shape) # torch.Size([2, 4, 64, 64])
        submask = self.convm(submask)
        # print(submask.shape) # torch.Size([2, 256, 64, 64])

        espa_in = fin * submask + subcontext * (1. - submask)
        # print(espa_in.shape) # torch.Size([2, 256, 64, 64])
        espa_out = self.ESP_Attn(espa_in)
        # print(espa_out.shape) # torch.Size([2, 256, 64, 64])

        out = x_ffc + espa_out
        out = self.kernel_pred(out, k1, white_level=1.0, rate=1)
        # print(out.shape) # torch.Size([2, 256, 64, 64])

        out = self.decoder(out)
        # print(out.shape) # torch.Size([2, 3, 256, 256])

        out = self.kernel_pred(out, k2, white_level=1.0, rate=1)

        out = (torch.tanh(out) + 1) / 2

        return out # torch.Size([2, 3, 256, 256])

if __name__ == '__main__':
    x = torch.rand(2, 4, 256, 256).cuda()
    m = torch.rand(2, 3, 256, 256).cuda()
    model = InpaintGenerator().cuda()
    out = model(x, m)
    print(out.shape)
