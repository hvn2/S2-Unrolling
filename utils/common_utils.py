import torch
import numpy as np
from torch.fft import fft2 as fft2
from torch.fft import ifft2 as ifft2

def hwc2chw(x):
    return(x.permute(2,0,1))
def chw2hwc(x):
    return(x.permute(1,2,0))

def sreCal(Xref,X):
    '''Calculate signal to reconstructed error (SRE) between reference image and reconstructed image
    Input: Xref, X: reference and reconstructed images in shape [h,w,d]
    Output: aSRE average SRE in dB
            SRE_vec: SRE of each band'''
    mSRE=0
    if len(Xref.shape)==3:
        Xref=Xref.reshape(Xref.shape[0]*Xref.shape[1],Xref.shape[2])
        X=X.reshape(X.shape[0]*X.shape[1],X.shape[2])
        SRE_vec=np.zeros((X.shape[1]))
        for i in range(X.shape[1]):
            SRE_vec[i]=10*np.log10(np.sum(Xref[:,i]**2)/np.sum((Xref[:,i]-X[:,i])**2))
        mSRE=np.mean(SRE_vec)
    else:
        mSRE=10*np.log10(np.sum(Xref**2)/np.sum((X-Xref)**2))
        # mSRE = sre.numpy()
    return mSRE

def gaussian_filter(N=15, sigma=2.0):
    n = (N - 1) / 2.0
    y, x = np.ogrid[-n:n + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def create_conv_kernel(nl, nc, dx=15, dy=15):
    # Create conv_kernel of size (dx,dy) with psf=sdf in fft domain for an image of size (nl,nc)
    middlel = dx // 2
    middlec = dy // 2
    d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
    mtf = np.array([.32, .26, .28, .24, .38, .34, .34, .26, .33, .26, .22, .23])
    sdf = d * np.sqrt(-2 * np.log(mtf) / np.pi ** 2)
    sdf[d == 1] = 0
    L = len(d)
    B = torch.zeros([L, nl, nc])
    for i in range(L):
        if d[i] == 1:
            B[i, 0, 0] = 1
        else:
            h = torch.from_numpy(gaussian_filter(N=dx, sigma=sdf[i]))
            B[i, nl//2-middlel:nl//2+middlel+1, nc//2-middlec:nc//2+middlec+1] = h
#             B[i, int(nl % 2 + (nl - dx) // 2):int(nl % 2 + (nl + dx) // 2),
#             int(nc % 2 + (nc - dy) // 2):int(nc % 2 + (nc + dy) // 2)] = h
            B[i, :, :] = torch.fft.fftshift(B[i, :, :])
    FBM = torch.fft.fft2(B)
    return FBM
def pad_shift_psf(psf, m, n, p):
    '''move psf to center and shift'''
    dx=psf.shape[0]//2
    dy=psf.shape[1]//2

    PSF=np.zeros([m,n,p])
    for i in range(p):
        PSF[m//2-dx:m//2+dx+1,n//2-dy:n//2+dy+1,i]=psf
    return fft.fftshift(PSF,axes=[0,1])
def AxS2(X,FBM):
    '''compute Ax by using FFT
    Inputs: X ground truth 12 S2 bands
            FBM: kernel in FFT domains
    Ouput: List of tensor is 12 LR S2 bands'''
    d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
    Xf = torch.real(torch.fft.ifft2(torch.fft.fft2(X) * FBM))
    Yim = [Xf[idx, ::ratio, ::ratio] for idx, ratio in enumerate(d)]
    return Yim #return a list
def ATxS2(Yim,FBM):
    '''upsampling by inserting zeros between samples and filtering
    input: Yim: list of 12 S2 bands, FBM: FFT of psf'''
    Xf=torch.zeros(len(Yim),Yim[2].shape[0],Yim[2].shape[0]).type_as(Yim[0])
    d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
    for idx, ratio in enumerate(d):
        Xf[idx, ::ratio, ::ratio]=Yim[idx]
    Xm = torch.real(torch.fft.ifft2(torch.fft.fft2(Xf) * torch.conj(FBM)))
    return Xm #return a tensor
def AATinvS2(Yim,FBM,cond):
    '''compute iverse(AAT) using polyphase trick
    input: Yim: list of LR 12 S2 bands, FBM: FFT of psf'''
    d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
    FBM0 = torch.real(ifft2(abs(FBM)**2)) #BBT
    X=[]
    for idx, ratio in enumerate(d):
        FBM0d=FBM0[idx, ::ratio, ::ratio] #0 th component of the polyphase decomposition BBT
        X.append(torch.real(ifft2(fft2(Yim[idx])/(fft2(FBM0d)+cond[idx]))))
    return X #return a list
def BPS2(Yim, FBM, cond): #input a list, return a tensor
    return ATxS2(AATinvS2(Yim,FBM,cond), FBM)