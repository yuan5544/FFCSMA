import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(net, init_type = 'normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )

        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        fm = self.conv1(data)

        if self.channel_att:
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att

        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att

        return fm


class KPN(nn.Module):
    def __init__(self, kernel_size=3, sep_conv=False, channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(KPN, self).__init__()
        self.upMode = upMode

        in_channel = 4
        out_channel = 64 * (kernel_size ** 2)

        self.conv1 = Basic(in_channel, 64, channel_att=channel_att, spatial_att=spatial_att)
        self.conv2 = Basic(64, 128, channel_att=channel_att, spatial_att=spatial_att)
        self.conv3 = Basic(128 + 128, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv4 = Basic(256, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256 + 512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256 + 256, 128, channel_att=channel_att, spatial_att=spatial_att)
        self.conv9 = Basic(128 + 64, 64, channel_att=channel_att, spatial_att=spatial_att)

        self.kernels = nn.Conv2d(256, out_channel, 1, 1, 0)

        out_channel_img = 3 * (kernel_size ** 2)
        self.core_img = nn.Conv2d(64, out_channel_img, 1, 1, 0)

    def forward(self, data_with_est, x):
        '''
        :param data_with_est: input
        :param x: x1
        :return:
        '''

        conv1 = self.conv1(data_with_est) # torch.Size([2, 64, 256, 256])
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2)) # torch.Size([2, 128, 128, 128])
        conv2 = torch.cat([conv2, x], dim=1) # torch.Size([2, 256, 128, 128])
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2)) # torch.Size([2, 256, 64, 64])

        kernels = self.kernels(conv3) # torch.Size([2, 576, 64, 64])
        kernels = kernels.unsqueeze(dim=0) # torch.Size([1, 2, 576, 64, 64])
        kernels = F.interpolate(input=kernels, size=(256*9, data_with_est.shape[-1]//4, data_with_est.shape[-2]//4), mode='nearest') # torch.Size([1, 2, 2304, 64, 64])
        kernels = kernels.squeeze(dim=0) # torch.Size([2, 2304, 64, 64])

        conv4 = self.conv4(conv3) # torch.Size([2, 512, 64, 64])
        conv7 = self.conv7(torch.cat([conv3, conv4], dim=1)) # torch.Size([2, 256, 64, 64])
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1)) # torch.Size([2, 128, 128, 128])
        conv9 = self.conv9(torch.cat([conv1, F.interpolate(conv8, scale_factor=2, mode=self.upMode)], dim=1)) # torch.Size([2, 128, 128, 128])
        core_img = self.core_img(conv9)

        '''
        torch.Size([2, 2304, 64, 64])
        torch.Size([2, 27, 256, 256])
        '''
        return kernels, core_img


if __name__ == '__main__':
    x = torch.rand(2, 4, 256, 256).cuda()
    m = torch.rand(2, 128, 128, 128).cuda()
    model = KPN().cuda()
    o1, o2 = model(x, m)
    print(o1.shape)
    print(o2.shape)