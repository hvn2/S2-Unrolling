import torch
import torch.nn as nn
from models.common import *
from models.skip import *
from utils.common_utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class s2_net(nn.Module):
    def __init__(self, in_channel=12, out_channel=12, num_channels_down=128, num_channels_up=128, num_channels_skip=4,
                 filter_size_down=3, filter_size_up=3, filter_skip_size=1):
        super().__init__()
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channel, num_channels_down, filter_size_down,
                      padding='same',padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            # nn.Softplus())
            nn.LeakyReLU(0.2, inplace=True))
        self.enc = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_down, filter_size_down,padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            # nn.Softplus())
            nn.LeakyReLU(0.2, inplace=True))

        self.skip = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_skip, filter_skip_size, padding ='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_skip),
            # nn.Softplus())
            nn.LeakyReLU(0.2, inplace=True))

        self.dc = nn.Sequential(
            nn.Conv2d((num_channels_skip + num_channels_up), num_channels_up, filter_size_up,padding='same',padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_up),
            # nn.Softplus())
            nn.LeakyReLU(0.2, inplace=True))
        self.out_layer = nn.Sequential(
            nn.Conv2d(num_channels_up, out_channel, 1, padding_mode='reflect'),
            nn.Sigmoid())
        self.conv = nn.Conv2d(num_channels_down,num_channels_down,filter_size_down, padding = 'same',padding_mode = 'reflect')
    def forward(self, inputs):
        '''inputs: x: a tensor is composed of 10 m, upsampled 20 and 60 m'''
        # encoder part
        y_en0 = self.enc0(inputs)
        y_en1 = self.enc(y_en0)
        y_en2 = self.enc(y_en1)
        y_en3 = self.enc(y_en2)
        y_en4 = self.enc(y_en3)
        # decoder part with skip connections
        y_dc0 = self.enc(y_en4)
        y_dc1 = self.dc(torch.cat((self.skip(y_en4), y_dc0), dim=1))
        y_dc2 = self.dc(torch.cat((self.skip(y_en3), y_dc1), dim=1))
        y_dc3 = self.dc(torch.cat((self.skip(y_en2), y_dc2), dim=1))
        y_dc4 = self.dc(torch.cat((self.skip(y_en1), y_dc3), dim=1))
        y_dc5 = self.dc(torch.cat((self.skip(y_en0), y_dc4), dim=1))

        out = self.out_layer(y_dc5)

        return out


class DataNet(nn.Module):
    '''Close form solution for the data term
    input: rho0: initial valued for rho '''

    def __init__(self,rho0=10.):
        super(DataNet, self).__init__()
        # self.rho = torch.nn.Parameter(torch.ones((12,1,1)), requires_grad = True)
        self.rho = rho0 * torch.ones((12, 1, 1)).to(device)

    def forward(self, Yim, zk1, uk, FBM):
        '''zk1, uk: are output of previous block
        Yim: LR image (list), FBM: kernel in FFT domain
        output: x'''
        rho = self.rho
        xtk = zk1 - uk
        bk = ATxS2(Yim, FBM) + rho * xtk
        xk = (bk - ATxS2(AATinvS2(AxS2(bk, FBM), FBM, rho), FBM)) / rho
        return xk
# def data(Yim,zk1,uk,FBM,rho):
#     xtk = zk1 - uk
#     bk = ATxS2(Yim, FBM) + rho * xtk
#     xk = (bk - ATxS2(AATinvS2(AxS2(bk, FBM), FBM, rho), FBM)) / rho
#     return xk

class Denoiser(nn.Module):
    '''A shallow denoiser with single conv layer and a LeakyReLU layer '''

    def __init__(self, in_filter=12,out_filter=12):
        super(Denoiser, self).__init__()
        self.denoise = nn.Sequential(
            nn.Conv2d(in_filter, out_filter, 3, padding='same',padding_mode='reflect'),
            # nn.BatchNorm2d(hid_filter),
            # nn.LeakyReLU(),
            #
            # nn.Conv2d(hid_filter, out_filter, 3, padding='same', padding_mode='reflect'),
            # # nn.BatchNorm2d(12),
            nn.LeakyReLU())

    def forward(self, inputs):
        return self.denoise(inputs)


class S2UnrollNet(nn.Module):
    """Unrolling network:
    data->denoiser->data->denoiser...
    an unrolling network composed of one iteration with a skip connection CNN for the denoiser tends to give best results"""
    def __init__(self,rho0=10.0):
        super(S2UnrollNet, self).__init__()
        self.data1 = DataNet(rho0=rho0)
        # self.data2 = DataNet(rho0=rho0)
        # self.data3 = DataNet(rho0=rho0)
        # self.data4 = DataNet(rho0=rho0)
        # self.data5 = DataNet(rho0=rho0)
        # self.data6 = DataNet(rho0=rho0)
        self.denoiser1 = skip()
        # self.denoiser2 = skip()
        # self.denoiser3 = skip()
        # self.denoiser4 = skip()
        # self.denoiser5 = skip()
        # self.denoiser6 = skip()

    def forward(self, x0, u0, Yim, FBM):
        ''' x0, u0, Yim, FBM are inputs of DataNet
        Output: x--> estimated image'''
        zt0 = x0 + u0
        z1 = self.denoiser1(zt0[None,:,:,:])
        x1 = self.data1(Yim,z1.squeeze(),u0,FBM)
        u1 = u0 + x1-z1.squeeze()
        # #
        # zt1 = x1 + u1
        # z2 = self.denoiser2(zt1[None,:,:,:])
        # x2 = self.data2(Yim,z2.squeeze(),u1,FBM)
        # u2 = u1 + x2-z2.squeeze()
        # # # # #
        # zt2 = x2 + u2
        # z3 = self.denoiser3(zt2[None, :, :, :])
        # x3 = self.data3(Yim, z3.squeeze(), u2, FBM)
        # u3 = u2 + x3 - z3.squeeze()
        #
        # zt3 = x3 + u3
        # z4 = self.denoiser4(zt3[None, :, :, :])
        # x4 = self.data4(Yim, z4.squeeze(), u3, FBM)
        # u4 = u3 + x4 - z4.squeeze()
        # # # # # #
        # zt4 = x4 + u4
        # z5 = self.denoiser5(zt4[None, :, :, :])
        # x5 = self.data5(Yim, z5.squeeze(), u4, FBM)
        # u5 = u4 + x5 - z5.squeeze()
        #
        # zt5 = x5 + u5
        # z6 = self.denoiser6(zt5[None, :, :, :])
        # x6 = self.data6(Yim, z6.squeeze(), u5, FBM)
        # u6 = u5 + x6 - z6.squeeze()

        return x1
def SURE_S2_loss(x, u, net, Yim, sigmas, FBM, cond, loss_type="sure"):
    '''MSE, BP, and SURE losses
    x: net input (1,c,m,n); Yim: observation ((1,c,m,n)); net: network; sigmas: sigmas for 10, 20 and 60 bands
    FBM: kernel in FFT domain; cond=1e-3; loss_type: "sure", "bp" and 'MSE'
    Output: loss, x'''
    # Compute MSE and BP losses
    xhat = net(x, u, Yim, FBM)
    yhat = AxS2(xhat.squeeze(), FBM)
    mse = 0.
    for j in range(len(Yim)):
        mse += torch.sum((yhat[j] - Yim[j]) ** 2)
    pxhat = BPS2(Yim, FBM, cond) - BPS2(yhat, FBM, cond)  # tensor
    loss_bp = torch.sum(pxhat ** 2)
    # Compute divergence term by Monte-Carlo SURE
    ep = 1e-5
    b = torch.randn(x.shape).to(device)
    xe=net(x + ep * b, u, Yim, FBM)
    oute = (xe - xhat) / ep
    houte = AxS2(oute.squeeze(), FBM)
    tmp = AATinvS2(houte, FBM, cond)  # a list
    tmp_sig = [tmp[i] * sigmas[i] ** 2 for i in range(len(tmp))]
    outep = BPS2(tmp_sig, FBM, cond)  # a tensor

    div = torch.sum(b * outep)
    if loss_type == "bp":
        return loss_bp, xhat.squeeze()
    elif loss_type == "sure":
        return loss_bp + 2 * div, xhat.squeeze()
    elif loss_type == 'mse':
        return mse, xhat.squeeze()
    else:
        print("none of loss used")