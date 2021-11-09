import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (Conv2d, ConvTranspose2d as ConvT2d, Sequential as Seq,
    ELU, ReLU, BatchNorm1d as BN, Linear as Lin, Identity as Id, ModuleList)
from src.GCN2 import GCN2Conv


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i-1], channels[i]), ELU(),
            (BN(channels[i]) if batch_norm==True else Id()))
        for i in range(1, len(channels))
    ])


class NetGCN2(nn.Module):
    def __init__(self, full_dim, reduced_dim, ideal_dim, ff_layers,
                 in_channels, num_out_channels, edge_index, res=0.2, theta=1.5):
        super().__init__()
        self.edge_index = edge_index
        self.full_dim = full_dim
        self.numF = in_channels

        # dynamics
        self.dyn_layers = nn.ModuleList()
        for idx in range(ff_layers):
            if idx == 0:
                self.dyn_layers.append(Lin(ideal_dim, 50))
            elif idx != ff_layers - 1:
                self.dyn_layers.append(Lin(50, 50))
            else:
                self.dyn_layers.append(Lin(50, reduced_dim))

        # encoder
        self.enc_layers = nn.ModuleList()
        for idx in range(num_out_channels):
            if idx == 0:
                self.enc_layers.append(
                    GCN2Conv(in_channels, res, node_dim=1, theta=theta,
                             cached=True, layer=idx+1))
            else:
                self.enc_layers.append(
                    GCN2Conv(in_channels, res, node_dim=1, theta=theta,
                             cached=True, layer=idx+1))
        self.enc_layers.append(Lin(full_dim, reduced_dim))

        # decoder
        self.dec_layers = nn.ModuleList()
        self.dec_layers.append(Lin(reduced_dim, full_dim))
        for idx in range(num_out_channels):
            if idx == 0:
                self.dec_layers.append(
                    GCN2Conv(in_channels, res, node_dim=1, theta=theta,
                             layer=num_out_channels+idx+1, cached=True))
            else:
                self.dec_layers.append(
                    GCN2Conv(in_channels, res, node_dim=1, theta=theta,
                        layer=num_out_channels+idx+1, cached=True))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encode(self, x):
        x0 = torch.clone(x)
        for i, layer in enumerate(self.enc_layers):
            if i != len(self.enc_layers) - 1:
                x = F.relu(layer(x, x0, self.edge_index))
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x

    def decode(self, x):
        num_layers = len(self.dec_layers)
        for i, layer in enumerate(self.dec_layers):
            if i == 0:
                x = layer(x).view(-1, self.full_dim, self.numF)
                x0 = torch.clone(x)
            else:
                x = F.relu(layer(x, x0, self.edge_index))
        return x

    def predict(self, x):
        for i, layer in enumerate(self.dyn_layers):
            if i < len(self.dyn_layers)-1:
                x = F.elu(layer(x))
            else: x = layer(x)
        return x

    # run autoencoder
    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out


class NetFCNN(torch.nn.Module):
    def __init__(self, num_features, reduced_dim, ideal_dim, ff_layers,
                 channels_list):
        super().__init__()
        channels_list.append(reduced_dim)
        self.full_dim = int(channels_list[0]/num_features)
        self.numF = num_features

        self.enc_layers = nn.ModuleList()
        self.enc_layers.append(MLP(channels_list[:-1]))
        self.enc_layers.append(Lin(channels_list[-2], channels_list[-1]))

        self.dec_layers = nn.ModuleList()
        self.dec_layers.append(MLP(channels_list[:0:-1]))
        self.dec_layers.append(Lin(channels_list[1], channels_list[0]))

        self.dyn_layers = nn.ModuleList()
        for idx in range(ff_layers):
            if idx == 0:
                self.dyn_layers.append(Lin(ideal_dim, 50))
            elif idx != ff_layers - 1:
                self.dyn_layers.append(Lin(50, 50))
            else:
                self.dyn_layers.append(Lin(50, reduced_dim))

    def encode(self, x):
        x = self.enc_layers[0](x.view(-1, self.numF*self.full_dim))
        return F.elu(self.enc_layers[1](x))

    def decode(self, x):
        x = self.dec_layers[0](x)
        return F.elu(self.dec_layers[1](x).view(-1, self.full_dim, self.numF))

    def forward(self, x):
        return self.decode(self.encode(x))

    def predict(self, x):
        for i, layer in enumerate(self.dyn_layers):
            x = F.elu(layer(x))
        return x


class NetCNN_Burgers(nn.Module):
    def __init__(self, reduced_dim, ff_layers):
        super().__init__()

        ## Dynamics network
        self.dyn_layers = nn.ModuleList()
        for idx in range(ff_layers):
            if idx == 0:
                self.dyn_layers.append(Lin(3, 50))
            elif idx != ff_layers - 1:
                self.dyn_layers.append(Lin(50, 50))
            else:
                self.dyn_layers.append(Lin(50, reduced_dim))

        # Encoder
        self.enc_layers = nn.ModuleList()
        self.enc_layers.append(Conv2d(1, 8, 5, padding=2))
        self.enc_layers.append(Conv2d(8, 16, 5, padding=2, stride=2))
        self.enc_layers.append(Conv2d(16, 32, 5, padding=2, stride=2))
        self.enc_layers.append(Conv2d(32, 64, 5, padding=2, stride=2))
        self.enc_layers.append(Lin(256, reduced_dim))

        # Decoder
        self.dec_layers = nn.ModuleList()
        self.dec_layers.append(Lin(reduced_dim, 256))
        self.dec_layers.append(ConvT2d(64, 64, 5, padding=2,
                               stride=2, output_padding=1))
        self.dec_layers.append(ConvT2d(64, 32, 5, padding=2,
                               stride=2, output_padding=1))
        self.dec_layers.append(ConvT2d(32, 16, 5, padding=2,
                               stride=2, output_padding=1))
        self.dec_layers.append(ConvT2d(16, 1, 5, padding=2))

    def encode(self, x):
        x = x.transpose(1, 2).reshape(-1, 1, 16, 16)
        for i, layer in enumerate(self.enc_layers):
            if i != len(self.enc_layers) - 1:
                x = F.elu(layer(x))
            else:
                x = x.view(-1, layer.weight.size(1))
                x = F.elu(layer(x))
        return x

    def decode(self, x):
        for i, layer in enumerate(self.dec_layers):
            if i == 0:
                x = F.elu(layer(x))
                x = x.view(-1, 64, 2, 2)
            else:
                x = F.elu(layer(x))
        return x.view(-1, 1, 256).transpose(1, 2)

    def predict(self, x):
        for i, layer in enumerate(self.dyn_layers):
            x = F.elu(layer(x))
        return x

    def forward(self, x):  ## runs encoder/decoder
        z = self.encode(x)
        xx = self.decode(z)
        return xx


class NetCNN_NSE(nn.Module):
    def __init__(self, reduced_dim, ff_layers):
        super().__init__()

        ## Dynamics network
        self.dyn_layers = nn.ModuleList()
        for idx in range(ff_layers):
            if idx == 0:
                self.dyn_layers.append(Lin(2, 50))
            elif idx != ff_layers - 1:
                self.dyn_layers.append(Lin(50, 50))
            else:
                self.dyn_layers.append(Lin(50, reduced_dim))

        # Encoder
        self.enc_layers = nn.ModuleList()
        self.enc_layers.append(Conv2d(2, 8, 5, padding=2, stride=2))
        self.enc_layers.append(Conv2d(8, 32, 5, padding=2, stride=2))
        self.enc_layers.append(Conv2d(32, 128, 5, padding=2, stride=2))
        self.enc_layers.append(Conv2d(128, 512, 5, padding=2, stride=2))
        self.enc_layers.append(Conv2d(512, 2048, 5, padding=2, stride=2))
        self.enc_layers.append(Lin(2*16384, reduced_dim))

        # Decoder
        self.dec_layers = nn.ModuleList()
        self.dec_layers.append(Lin(reduced_dim, 2*16384))
        self.dec_layers.append(ConvT2d(2048, 512, 5, padding=2,
                               stride=2, output_padding=1))
        self.dec_layers.append(ConvT2d(512, 128, 5, padding=2,
                               stride=2, output_padding=1))
        self.dec_layers.append(ConvT2d(128, 32, 5, padding=2,
                               stride=2, output_padding=1))
        self.dec_layers.append(ConvT2d(32, 8, 5, padding=2,
                               stride=2, output_padding=1))
        self.dec_layers.append(ConvT2d(8, 2, 5, padding=2,
                               stride=2, output_padding=1))

    def encode(self, x):
        x = x.transpose(1, 2).reshape(-1, 2, 128, 128)
        for i, layer in enumerate(self.enc_layers):
            if i != len(self.enc_layers) - 1:
                x = F.elu(layer(x))
            else:
                x = x.view(-1, layer.weight.size(1))
                x = F.elu(layer(x))
        return x

    def decode(self, x):
        for i, layer in enumerate(self.dec_layers):
            if i == 0:
                x = F.elu(layer(x))
                x = x.view(-1, 2048, 4, 4)
            else:
                x = F.elu(layer(x))
        return x.view(-1, 2, 16384).transpose(1, 2)

    def predict(self, x):
        for i, layer in enumerate(self.dyn_layers):
            x = F.elu(layer(x))
        return x

    def forward(self, x):  ## runs encoder/decoder
        z = self.encode(x)
        xx = self.decode(z)
        return xx


class NetCNN_Heat(nn.Module):
    def __init__(self, reduced_dim, ff_layers):
        super().__init__()

        ## Dynamics network
        self.dyn_layers = nn.ModuleList()
        for idx in range(ff_layers):
            if idx == 0:
                self.dyn_layers.append(Lin(3, 50))
            elif idx != ff_layers - 1:
                self.dyn_layers.append(Lin(50, 50))
            else:
                self.dyn_layers.append(Lin(50, reduced_dim))

        # Encoder
        self.enc_layers = nn.ModuleList()
        self.enc_layers.append(Conv2d(1, 4, 5, padding=2, stride=2))
        self.enc_layers.append(Conv2d(4, 16, 5, padding=2, stride=2))
        self.enc_layers.append(Conv2d(16, 64, 5, padding=2, stride=2))
        self.enc_layers.append(Conv2d(64, 256, 5, padding=2, stride=2))
        self.enc_layers.append(Lin(4096, reduced_dim))

        # Decoder
        self.dec_layers = nn.ModuleList()
        self.dec_layers.append(Lin(reduced_dim, 4096))
        self.dec_layers.append(ConvT2d(256, 64, 5, padding=2,
                               stride=2, output_padding=1))
        self.dec_layers.append(ConvT2d(64, 16, 5, padding=2,
                               stride=2, output_padding=1))
        self.dec_layers.append(ConvT2d(16, 4, 5, padding=2,
                               stride=2, output_padding=1))
        self.dec_layers.append(ConvT2d(4, 1, 5, padding=2,
                               stride=2, output_padding=1))

    def encode(self, x):
        x = x.transpose(1, 2).reshape(-1, 1, 64, 64)
        for i, layer in enumerate(self.enc_layers):
            if i != len(self.enc_layers) - 1:
                x = F.elu(layer(x))
            else:
                x = x.view(-1, layer.weight.size(1))
                x = F.elu(layer(x))
        return x

    def decode(self, x):
        for i, layer in enumerate(self.dec_layers):
            if i == 0:
                x = F.elu(layer(x))
                x = x.view(-1, 256, 4, 4)
            else:
                x = F.elu(layer(x))
        return x.view(-1, 1, 4096).transpose(1, 2)

    def predict(self, x):
        for i, layer in enumerate(self.dyn_layers):
            x = F.elu(layer(x))
        return x

    def forward(self, x):  ## runs encoder/decoder
        z = self.encode(x)
        xx = self.decode(z)
        return xx
