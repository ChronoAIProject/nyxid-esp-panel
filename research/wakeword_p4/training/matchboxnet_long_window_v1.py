"""Separate 44800 past-PCM adapter; original short-window module stays frozen."""
import torch
from matchboxnet_pretrained_v1 import MatchboxNet, LogMFCC, FRONTEND_KEYS, load as load_original
WINDOW=44800
FRAMES=280

class LongMatchboxNet(MatchboxNet):
    def forward(self,features):
        if features.ndim!=3 or tuple(features.shape[1:])!=(64,FRAMES):
            raise ValueError('Expected complete [B,64,280] MFCC context')
        return self.decoder.decoder_layers(self.encoder.encoder(features).mean(dim=2))

class LongLogMFCC(LogMFCC):
    def forward(self,pcm):
        if pcm.ndim!=2 or pcm.shape[1]!=WINDOW or pcm.dtype!=torch.float32:
            raise ValueError('Expected complete 44800-sample float32 past-PCM window')
        spectrum=torch.stft(pcm,n_fft=512,hop_length=160,win_length=400,window=self.window,center=True,pad_mode='reflect',normalized=False,onesided=True,return_complex=True)
        mel=torch.matmul(spectrum.abs().square().transpose(1,2),self.mel).transpose(1,2)
        mfcc=torch.matmul(torch.log(mel+1e-6).transpose(1,2),self.dct).transpose(1,2)
        assert mfcc.shape[2]==FRAMES+1
        return mfcc[:,:,:FRAMES]

def load():
    front,old=load_original()
    state={key:value for key,value in zip(FRONTEND_KEYS,[front.dct,front.window,front.mel],strict=True)}
    model=LongMatchboxNet();model.load_state_dict(old.state_dict(),strict=True)
    return LongLogMFCC(state),model
