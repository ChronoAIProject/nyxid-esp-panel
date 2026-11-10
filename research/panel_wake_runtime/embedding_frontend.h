#pragma once
#include <algorithm>
#include <cmath>
#include "native-coefficients.h"

struct NativeScratch {
    float real[512], imag[512], power[257], features[256];
};

// Each call is exactly 480 preceding samples plus 1280 new samples at 16 kHz.
// Coefficients and blockwise 80 dB floor match the frozen source frontend.
static void native_frontend(const int16_t *pcm, NativeScratch &scratch) {
    float highest=-INFINITY;
    for(unsigned frame=0;frame<8;++frame) {
        for(unsigned sample=0;sample<512;++sample) {
            unsigned reversed=0,value=sample;
            for(unsigned bit=0;bit<9;++bit){reversed=(reversed<<1)|(value&1);value>>=1;}
            scratch.real[reversed]=float(pcm[frame*160+sample])*fft_window[sample];
            scratch.imag[reversed]=0;
        }
        for(unsigned length=2;length<=512;length*=2) {
            const unsigned half=length/2,stride=512/length;
            for(unsigned base=0;base<512;base+=length) {
                for(unsigned k=0;k<half;++k) {
                    const unsigned a=base+k,b=a+half,w=k*stride;
                    const float real=fft_cos[w]*scratch.real[b]-fft_sin[w]*scratch.imag[b];
                    const float imag=fft_cos[w]*scratch.imag[b]+fft_sin[w]*scratch.real[b];
                    scratch.real[b]=scratch.real[a]-real;
                    scratch.imag[b]=scratch.imag[a]-imag;
                    scratch.real[a]+=real;scratch.imag[a]+=imag;
                }
            }
        }
        for(unsigned k=0;k<257;++k)
            scratch.power[k]=scratch.real[k]*scratch.real[k]+scratch.imag[k]*scratch.imag[k];
        for(unsigned channel=0;channel<32;++channel) {
            float sum=0;
            for(unsigned k=mel_offsets[channel];k<mel_offsets[channel+1];++k)
                sum+=scratch.power[mel_starts[channel]+k-mel_offsets[channel]]*mel_weights[k];
            const float value=log10f(std::max(sum,1e-10f))+2;
            scratch.features[frame*32+channel]=value;highest=std::max(highest,value);
        }
    }
    for(auto &value:scratch.features)value=std::max(value,highest-8);
}

static int native_quantize(float value) {
    return std::max(-128,std::min(127,int(std::nearbyint(value/input_scale))+input_zero));
}
