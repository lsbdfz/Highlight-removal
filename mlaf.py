import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt

def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin, pmax = np.min(img), np.max(img)
        img = (img - pmin) / (pmax - pmin + 1e-6)
        plt.imshow(img, cmap='gray')
        print(f"{i}/{width*height}")
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print(f"time: {time.time() - tic:.3f}s")

def exception_handler(predict_func):
    def wrapper(model, *args, **kwargs):
        try:
            return predict_func(model, *args, **kwargs)
        except RuntimeError:
            print("image is too large")
            exit()
    return wrapper

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.features(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=0, AF=nn.ReLU):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            AF()
        )
    def forward(self, x):
        return self.features(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=scale_factor, stride=scale_factor),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.features(x)

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (g * 0.1 for g in grad_input)
        grad_output = (g * 0.1 for g in grad_output)

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2,
                         (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2,
                         (self.kernel_size - 1) // 2 + 1)
        )
        p_n = torch.cat([p_n_x.flatten(), p_n_y.flatten()], dim=0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride)
        )
        p_0_x = p_0_x.flatten().view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = p_0_y.flatten().view(1, 1, h, w).repeat(1, N, 1, 1)
        return torch.cat([p_0_x, p_0_y], dim=1).type(dtype)

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        return p_0 + p_n + offset

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous()
        index = index.view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_ = []
        for s in range(0, N, ks):
            x_.append(x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks))
        x_offset = torch.cat(x_, dim=-1)
        x_offset = x_offset.view(b, c, h * ks, w * ks)
        return x_offset

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        dtype = offset.data.type()
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype)
        p = p.permute(0, 2, 3, 1)
        N = offset.size(1) // 2
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                          torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                          torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p_ = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1),
                        torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
        g_lt = ((1 + (q_lt[..., :N].type_as(p_) - p_[..., :N])) *
                (1 + (q_lt[..., N:].type_as(p_) - p_[..., N:])))
        g_rb = ((1 - (q_rb[..., :N].type_as(p_) - p_[..., :N])) *
                (1 - (q_rb[..., N:].type_as(p_) - p_[..., N:])))
        g_lb = ((1 + (q_lb[..., :N].type_as(p_) - p_[..., :N])) *
                (1 - (q_lb[..., N:].type_as(p_) - p_[..., N:])))
        g_rt = ((1 - (q_rt[..., :N].type_as(p_) - p_[..., :N])) *
                (1 + (q_rt[..., N:].type_as(p_) - p_[..., N:])))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = (g_lt.unsqueeze(dim=1) * x_q_lt +
                    g_rb.unsqueeze(dim=1) * x_q_rb +
                    g_lb.unsqueeze(dim=1) * x_q_lb +
                    g_rt.unsqueeze(dim=1) * x_q_rt)
        if self.modulation:
            m = m.permute(0, 2, 3, 1).unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        x_offset = self._reshape_x_offset(x_offset, self.kernel_size)
        out = self.conv(x_offset)
        return out

class SnakeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, modulation=True):
        super().__init__()
        self.deform_conv = DeformConv2d(in_channels, out_channels,
                                        kernel_size=kernel_size, padding=padding,
                                        modulation=modulation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, mask):
        x = self.deform_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x, mask

class PartialConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, dilation=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_I = nn.Conv2d(in_channels, out_channels,
                                kernel_size, padding=padding, dilation=dilation)
        self.conv_M = nn.Conv2d(1, 1, kernel_size,
                                padding=padding, dilation=dilation, bias=False)
        nn.init.constant_(self.conv_M.weight, 1.0)
        self.conv_M.requires_grad_(False)
    def forward(self, x, M):
        M = self.conv_M(M)
        index = (M == 0)
        M[index] = 1
        x = self.conv_I(M * x)
        x = F.relu(self.bn(x / M))
        M = M.masked_fill(index, 0)
        return x, M

class CoTAttention(nn.Module):
    def __init__(self, dim=3, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=3, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, (2 * dim) // factor, 1, bias=False),
            nn.BatchNorm2d((2 * dim) // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d((2 * dim) // factor, kernel_size * kernel_size * dim, 1)
        )
    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)
        v = self.value_embed(x).view(bs, c, -1)
        y = torch.cat([k1, x], dim=1)
        att = self.attention_embed(y)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2).view(bs, c, -1)
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)
        return k1 + k2

def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x

def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x

class SplitAttention(nn.Module):
    def __init__(self, channel=3, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.view(b, k, -1, c)
        a = x_all.sum(dim=1).sum(dim=1)
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))
        hat_a = hat_a.view(b, self.k, c)
        bar_a = self.softmax(hat_a)
        attention = bar_a.unsqueeze(-2)
        out = attention * x_all
        out = out.sum(dim=1).view(b, h, w, c)
        return out

class S2Attention(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()
    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], dim=1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x

class bingxing(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.cot = CoTAttention(dim=dim)
        self.s2 = S2Attention(channels=dim)
    def forward(self, x):
        a = self.cot(x)
        b = self.s2(x)
        return a + b

class lzfBlock(nn.Module):
    def __init__(self, channels=3, name=''):
        super().__init__()
        self.channels = channels
        self.bingxing = bingxing(dim=channels)
        self.conv5 = nn.Conv2d(64, channels, 1)
        self.conv4 = nn.Conv2d(32 + channels, channels * 2, 1)
        self.conv3 = nn.Conv2d(16 + channels * 2, channels * 3, 1)
        self.conv2 = nn.Conv2d(8 + channels * 3, channels * 4, 1)
        self.conv1 = nn.Conv2d(4 + channels * 4, channels * 5, 1)
    def forward(self, x1, x2, x3, x4, x5):
        x = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv5(x)
        x = self.bingxing(x)
        x = torch.cat([x, x4], dim=1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = torch.cat([x, x3], dim=1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv3(x)
        x = torch.cat([x, x2], dim=1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = torch.cat([x, x1], dim=1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        return x

class MlafNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = EncoderBlock(3, 4, 3, padding=1)
        self.encoder2 = EncoderBlock(4, 8, 3, padding=1)
        self.encoder3 = EncoderBlock(8, 16, 3, padding=1)
        self.encoder4 = EncoderBlock(16, 32, 3, padding=1)
        self.encoder5 = EncoderBlock(32, 64, 3, padding=1)
        self.decoder5 = DecoderBlock(64, 32, scale_factor=2)
        self.decoder4 = DecoderBlock(64, 16, scale_factor=2)
        self.decoder3 = DecoderBlock(32, 8, scale_factor=2)
        self.decoder2 = DecoderBlock(16, 4, scale_factor=2)
        self.decoder1 = DecoderBlock(8, 1, scale_factor=2)
        self.mlaf = lzfBlock()
        self.M_conv = nn.Sequential(
            ConvBlock(16, 8, 3, padding=1),
            ConvBlock(8, 4, 3, padding=1),
            ConvBlock(4, 1, 3, padding=1, AF=nn.Sigmoid)
        )
        self.S_conv = nn.Sequential(
            ConvBlock(17, 8, 3, padding=1),
            ConvBlock(8, 3, 3, padding=1)
        )
        self.D_conv1 = PartialConvBlock(19, 13, 5, padding=2)
        self.D_conv2 = PartialConvBlock(13, 8, 5, padding=2)
        self.D_conv3 = PartialConvBlock(8, 3, 5, padding=2)

    def forward(self, I):
        x1 = self.encoder1(I)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        x_mlaf = self.mlaf(x1, x2, x3, x4, x5)
        x6 = torch.cat([self.decoder5(x5), x4], dim=1)
        x7 = torch.cat([self.decoder4(x6), x3], dim=1)
        x8 = torch.cat([self.decoder3(x7), x2], dim=1)
        x9 = torch.cat([self.decoder2(x8), x1], dim=1)
        x10 = torch.cat([self.decoder1(x9), x_mlaf], dim=1)
        M = self.M_conv(x10)
        S = self.S_conv(torch.cat([x10, M], dim=1))
        D, M_ = self.D_conv1(torch.cat([x10, I - M * S], dim=1), 1 - M)
        D, M_ = self.D_conv2(D, M_)
        D, M_ = self.D_conv3(D, M_)
        return M, S, D

    @exception_handler
    def predict(self, image: Image.Image, use_gpu=True):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        w, h = image.size
        w_padded = (w // 32 + (w % 32 != 0)) * 32
        h_padded = (h // 32 + (h % 32 != 0)) * 32
        image_padded = cv2.copyMakeBorder(
            np.array(image), 0, h_padded - h, 0, w_padded - w, cv2.BORDER_REFLECT
        )
        device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        self.to(device)
        inp = T.ToTensor()(image_padded).unsqueeze(0).to(device)
        M, S, D = self(inp)
        M_pil = T.ToPILImage()(M.cpu().ge(0.5).float().squeeze())
        S_pil = T.ToPILImage()(S.cpu().squeeze())
        D_pil = T.ToPILImage()(D.cpu().squeeze())
        M_pil = M_pil.crop((0, 0, w, h))
        S_pil = S_pil.crop((0, 0, w, h))
        D_pil = D_pil.crop((0, 0, w, h))
        return M_pil, S_pil, D_pil

    def remove_specular(self, image: Image.Image):
        return self.predict(image)[-1]
