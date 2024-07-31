import torch.nn as nn


class Ex_KV(nn.Module):
    def __init__(self, dim, mlp_factor, bias_qkv=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_factor, 1, bias=bias_qkv),
            nn.GELU(),
            nn.Conv1d(dim * mlp_factor, dim, 1, bias=bias_qkv),
        )

    def forward(self, x):
        return self.net(x)


class ESP_Attn(nn.Module):
    def __init__(self, in_channels, in_width, in_height, token_facter):
        super().__init__()
        self.height = in_height
        self.width = in_width
        self.channels = in_channels
        self.norm = nn.LayerNorm(in_channels)
        self.ex_k = Ex_KV(in_height, token_facter, bias_qkv=False)
        self.ex_v = Ex_KV(in_width, token_facter, bias_qkv=False)

    def forward(self, x):
        b = x.shape[0]
        x = x.permute(0, 3, 2, 1)  # b c h w -> b w h c
        x = self.norm(x)
        x = x.view(b * self.width, self.height, self.channels)  # b w h c -> (b w) h c
        x = self.ex_k(x)
        x = x.view(b, self.width, self.height, self.channels).transpose(1, 2).flatten(0, 1)  # (b w) h c -> (b h) w c
        x = self.ex_v(x)
        x = x.view(b, self.height, self.width, self.channels).permute(0, 3, 1, 2)  # (b h) w c -> b c h w
        return x
