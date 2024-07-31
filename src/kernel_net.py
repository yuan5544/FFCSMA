import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelConv(nn.Module):
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, 3, height, width)
            core_out[K] = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0, rate=1):
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if not img_stack:
                padding_num = (K//2) * rate
                frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])
                for i in range(0, K):
                    for j in range(0, K):
                        img_stack.append(frame_pad[..., i*rate:i*rate + height, j*rate:j*rate + width])
                img_stack = torch.stack(img_stack, dim=2)
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]

            pred_img.append(torch.sum(
                core[K].mul(img_stack), dim=2, keepdim=False
            ))

        pred_img = torch.stack(pred_img, dim=0)

        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)

        pred_img_i = pred_img_i.squeeze(2)

        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias

        pred_img_i = pred_img_i / white_level

        return pred_img_i