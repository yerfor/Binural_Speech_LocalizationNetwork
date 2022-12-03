import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils import Net


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, dilation=1):
        '''
        :param ch_in: (int) input channels
        :param ch_out: (int) output channels
        :param z_dim: (int) dimension of the weight-generating input
        :param kernel_size: (int) size of the filter
        :param dilation: (int) dilation
        '''
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = nn.Conv1d(ch_in, ch_out, kernel_size, dilation=dilation)
        self.residual = nn.Conv1d(ch_out, ch_out, kernel_size=1)
        self.residual.weight.data.uniform_(-np.sqrt(6.0/ch_out), np.sqrt(6.0/ch_out))
        self.skip = nn.Conv1d(ch_out, ch_out, kernel_size=1)
        self.skip.weight.data.uniform_(-np.sqrt(6.0/ch_out), np.sqrt(6.0/ch_out))
        if not ch_in == ch_out:
            self.equalize_channels = nn.Conv1d(ch_in, ch_out, kernel_size=1)
            self.equalize_channels.weight.data.uniform_(-np.sqrt(6.0 / ch_in), np.sqrt(6.0 / ch_in))

    def forward(self, x):
        '''
        :param x: input signal as a B x ch_in x T tensor
        :param z: weight-generating input as a B x z_dim x K tensor (K s.t. T is a multiple of K)
        :return: output: B x ch_out x T tensor as layer output
                 skip: B x ch_out x T tensor as skip connection output
        '''
        y = self.conv(x)
        padding = self.dilation * (self.kernel_size - 1)
        y = F.pad(y, [padding, 0])

        y = th.sin(y)
        # residual and skip
        residual = self.residual(y)
        if not self.ch_in == self.ch_out:
            x = self.equalize_channels(x)
        skip = self.skip(y)
        return (residual + x) / 2, skip

    def receptive_field(self):
        return (self.kernel_size - 1) * self.dilation + 1


class ConvWavenet(nn.Module):
    def __init__(self, channels=64, blocks=3, layers_per_block=10, conv_len=2):
        super().__init__()
        self.layers = []
        self.rectv_field = 1
        for b in range(blocks):
            for l in range(layers_per_block):
                self.layers += [ConvBlock(channels, channels, kernel_size=conv_len, dilation=2**l)]
                self.rectv_field += self.layers[-1].receptive_field() - 1
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        '''
        :param x: input signal as a B x channels x T tensor
        :param z: weight-generating input as a B x z_dim z K tensor (K = T / 400)
        :return: x: output signal as a B x channels x T tensor
                 skips: skip signal for each layer as a list of B x channels x T tensors
        '''
        skips = []
        for layer in self.layers:
            x, skip = layer(x)
            skips += [skip]
        return x, skips

    def receptive_field(self):
        return self.rectv_field


class WaveoutBlock(nn.Module):
    def __init__(self, channels, out_channels=2):
        super().__init__()
        self.first = nn.Conv1d(channels, channels, kernel_size=1)
        self.first.weight.data.uniform_(-np.sqrt(6.0 / channels), np.sqrt(6.0 / channels))
        self.second = nn.Conv1d(channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = th.sin(self.first(x))
        b,c,t = x.shape
        x = x.reshape([b,c,t//400, 400]).mean(dim=-1) # downsample to view sample rate!
        return self.second(x)


class LocalizationNetwork(Net):
    def __init__(self,
                 view_dim=7,
                 wavenet_blocks=3,
                 layers_per_block=10,
                 wavenet_channels=64,
                 model_name='localization_network',
                 use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.input = nn.Conv1d(2, wavenet_channels, kernel_size=1)
        self.input.weight.data.uniform_(-np.sqrt(6.0 / 2), np.sqrt(6.0 / 2))
        self.conv_wavenet = ConvWavenet(wavenet_channels, wavenet_blocks, layers_per_block)
        self.output_net = nn.ModuleList([WaveoutBlock(wavenet_channels, view_dim)
                                        for _ in range(wavenet_blocks*layers_per_block)])
        if self.use_cuda:
            self.cuda()

    def forward(self, bina):
        '''
        :param bina: the input signal as a B x 2 x T tensor
        :return pred_view: the receiver/transmitter position as a B x 7 x T//400 tensor
        '''
        x = self.input(bina)
        _, skips = self.conv_wavenet(x)
        # collect output and skips after each layer
        x = []
        for k in range(len(skips), 0, -1):
            y = th.mean(th.stack(skips[:k], dim=0), dim=0)
            y = self.output_net[k-1](y)
            x += [y]
        return {"output": x[0], "intermediate": x[1:]}

    def receptive_field(self):
        return self.conv_wavenet.receptive_field()



if __name__ == '__main__':
    net = LocalizationNetwork()
    x = th.rand([1, 2, 9600]).cuda()
    y = net(x)
    print(" ")