"""Small, explicit adapter for the pinned NGC MatchboxNet 3x1x64 checkpoint.

Fixed full feature windows only. This is NOT a causal streaming conversion or a
claim of numerical equivalence with an unidentified original training runtime.
NeMo rc1 Jasper/ConvASR and torchaudio0.7.2 log-MFCC semantics are documented in
the source audit. All checkpoint neural tensors must load strictly.
"""
import hashlib
import io
from pathlib import Path
import tarfile
import torch
from torch import nn

ARCHIVE=Path('.local/wakeword/compact-pretrained-feasibility-v1/matchboxnet3x1x64-v2-1.0.0rc1.nemo')
ARCHIVE_SHA='0697bff0f8b7cb77052a2c2204680b1f948c646664406a4121944880d0a54286'
FRONTEND_KEYS=('preprocessor.featurizer.dct_mat','preprocessor.featurizer.MelSpectrogram.spectrogram.window','preprocessor.featurizer.MelSpectrogram.mel_scale.fb')

class Conv(nn.Module):
    def __init__(self,channels,filters,kernel=1,dilation=1,groups=1):
        super().__init__()
        self.conv=nn.Conv1d(channels,filters,kernel,padding=dilation*(kernel-1)//2,dilation=dilation,groups=groups,bias=False)
    def forward(self,x):return self.conv(x)

class Block(nn.Module):
    def __init__(self,channels,filters,kernel,dilation=1,separable=True,residual=False):
        super().__init__()
        conv=[Conv(channels,channels,kernel,dilation,channels),Conv(channels,filters)] if separable else [Conv(channels,filters,kernel,dilation)]
        self.mconv=nn.ModuleList(conv+[nn.BatchNorm1d(filters,eps=.001,momentum=.1)])
        self.res=nn.ModuleList([nn.ModuleList([Conv(channels,filters),nn.BatchNorm1d(filters,eps=.001,momentum=.1)])]) if residual else None
        self.mout=nn.Sequential(nn.ReLU(),nn.Dropout(0.))
    def forward(self,x):
        out=x
        for layer in self.mconv:out=layer(out)
        if self.res is not None:
            for branch in self.res:
                value=x
                for layer in branch:value=layer(value)
                out=out+value
        return self.mout(out)

class MatchboxNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Module()
        self.encoder.encoder=nn.Sequential(Block(64,128,11),Block(128,64,13,residual=True),Block(64,64,15,residual=True),Block(64,64,17,residual=True),Block(64,128,29,dilation=2),Block(128,128,1,separable=False))
        self.decoder=nn.Module()
        self.decoder.decoder_layers=nn.Sequential(nn.Linear(128,35))
    def forward(self,features):
        if features.ndim!=3 or features.shape[1]!=64 or features.shape[2]!=128:
            raise ValueError('This feasibility adapter requires full [batch,64,128] windows, no variable-length mask')
        return self.decoder.decoder_layers(self.encoder.encoder(features).mean(dim=2))

class LogMFCC(nn.Module):
    def __init__(self,state):
        super().__init__()
        for name,key,shape in zip(['dct','window','mel'],FRONTEND_KEYS,[(64,64),(400,),(257,64)],strict=True):
            value=state[key];assert tuple(value.shape)==shape and value.dtype==torch.float32
            self.register_buffer(name,value.detach().clone())
    def forward(self,pcm):
        if pcm.ndim!=2 or pcm.shape[1]!=20480 or pcm.dtype!=torch.float32:
            raise ValueError('Fixed 20480-sample float32 PCM window required')
        spectrum=torch.stft(pcm,n_fft=512,hop_length=160,win_length=400,window=self.window,center=True,pad_mode='reflect',normalized=False,onesided=True,return_complex=True)
        power=spectrum.abs().square()
        mel=torch.matmul(power.transpose(1,2),self.mel).transpose(1,2)
        mfcc=torch.matmul(torch.log(mel+1e-6).transpose(1,2),self.dct).transpose(1,2)
        # Explicit proposed fixed-window convention: 129 centered frames -> first
        # 128. This is not the original train/eval crop augmentation or a claim
        # that independently centered windows equal a continuous frontend.
        assert mfcc.shape[2]==129
        return mfcc[:,:,:128]

def load():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==ARCHIVE_SHA
    with tarfile.open(ARCHIVE,'r:*') as tar:raw=tar.extractfile('./model_weights.ckpt').read()
    state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    model=MatchboxNet();expected=set(model.state_dict())|set(FRONTEND_KEYS)
    if set(state)!=expected:raise ValueError('Unexpected/missing original checkpoint tensors')
    model.load_state_dict({k:v for k,v in state.items() if k not in FRONTEND_KEYS},strict=True)
    return LogMFCC(state),model
