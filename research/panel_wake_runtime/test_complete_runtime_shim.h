// Test-only host doubles. No FFT, model inference, ESP hardware or audio IO.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace shim {
using Group=std::array<int8_t,120>;
inline std::vector<Group> groups;
inline bool bad_shape=false, bad_quant=false;
inline int fail_at=-1;
inline uint8_t raw(const Group &group) {
    unsigned result=0;
    for(auto value:group)result=(result*17+uint8_t(value))&255;
    return uint8_t(result);
}
inline uint16_t feature(const int16_t *window,unsigned channel) {
    int64_t sum=0;
    for(unsigned j=channel;j<480;j+=40)sum+=int64_t(window[j])*(j+1);
    return uint16_t(((sum%19+19)%19)*32);
}
inline void reset(){groups.clear();bad_shape=bad_quant=false;fail_at=-1;}
}
#define MALLOC_CAP_INTERNAL 1
#define MALLOC_CAP_8BIT 2
inline void *heap_caps_aligned_alloc(size_t,size_t size,int){return malloc(size);}
inline int64_t esp_timer_get_time(){static int64_t time=0;return ++time;}
inline unsigned uxTaskGetStackHighWaterMark(void *){return 10000;}
#define ESP_LOGI(...) ((void)0)
#define ESP_LOGE(...) ((void)0)

using det_mode_t=int;
using wakenet_state_t=int;
struct model_iface_data_t {};
struct esp_wn_iface_t {
    model_iface_data_t *(*create)(const void *,det_mode_t);
    void *get_start_point;
    int (*get_samp_chunksize)(model_iface_data_t *);
    int (*get_channel_num)(model_iface_data_t *);
    int (*get_samp_rate)(model_iface_data_t *);
    void *get_word_num,*get_word_name,*set_det_threshold,*reset_det_threshold,
         *get_det_threshold,*get_triggered_channel,*get_vol_gain;
    wakenet_state_t (*detect)(model_iface_data_t *,int16_t *);
    void *detect_mfcc,*get_mfcc_data,*clean;
    void (*destroy)(model_iface_data_t *);
};

enum {kTfLiteOk=0,kTfLiteError=1,kTfLiteInt8=9,kTfLiteUInt8=3,TFLITE_SCHEMA_VERSION=3};
struct TfLiteIntArray {int size;int data[4];};
struct TfLiteTensor {
    int type;size_t bytes;
    struct {float scale;int zero_point;} params;
    union {int8_t *int8;uint8_t *uint8;} data;
    TfLiteIntArray *dims;
};
namespace tflite {
struct Model {int version()const{return TFLITE_SCHEMA_VERSION;}};
inline const Model *GetModel(const uint8_t *){static Model model;return &model;}
struct MicroAllocator {
    static MicroAllocator *Create(uint8_t *,size_t){static MicroAllocator value;return &value;}
};
struct MicroResourceVariables {
    static MicroResourceVariables *Create(MicroAllocator *,int){static MicroResourceVariables value;return &value;}
};
template<unsigned N>struct MicroMutableOpResolver {
#define OP(name) int Add##name(){return kTfLiteOk;}
    OP(CallOnce) OP(VarHandle) OP(ReadVariable) OP(Reshape) OP(Concatenation)
    OP(StridedSlice) OP(AssignVariable) OP(Conv2D) OP(DepthwiseConv2D) OP(SplitV)
    OP(FullyConnected) OP(Logistic) OP(Quantize) OP(ExpandDims)
#undef OP
};
struct MicroInterpreter {
    int8_t buffer[120]{};uint8_t result=0;
    TfLiteIntArray input_dims{3,{1,3,40}},output_dims{2,{1,1}};
    TfLiteTensor in{kTfLiteInt8,120,{.25f,-90},{.int8=buffer},&input_dims};
    TfLiteTensor out{kTfLiteUInt8,1,{1.f/256,0},{.uint8=&result},&output_dims};
    template<class Ops>MicroInterpreter(const Model *,Ops &,uint8_t *,size_t,MicroResourceVariables *){
        if(shim::bad_shape)input_dims.data[1]=4;
        if(shim::bad_quant)out.params.scale=1.f/255;
    }
    int AllocateTensors(){return kTfLiteOk;}
    TfLiteTensor *input(int){return &in;}
    TfLiteTensor *output(int){return &out;}
    unsigned arena_used_bytes(){return 1234;}
    int Invoke(){
        if(int(shim::groups.size())==shim::fail_at)return kTfLiteError;
        shim::Group group;
        std::copy_n(buffer,120,group.begin());shim::groups.push_back(group);
        result=shim::raw(group);return kTfLiteOk;
    }
};
}

struct FrontendConfig {
    struct {int size_ms,step_size_ms;} window;
    struct {int num_channels,lower_band_limit,upper_band_limit;} filterbank;
    struct {int smoothing_bits;double even_smoothing,odd_smoothing,min_signal_remaining;} noise_reduction;
    struct {int enable_pcan;double strength;int offset,gain_bits;} pcan_gain_control;
    struct {int enable_log,scale_shift;} log_scale;
};
struct FrontendState {std::vector<int16_t> pcm;uint16_t features[40]{};size_t next=480;};
struct FrontendOutput {const uint16_t *values;size_t size;};
extern "C" {
inline void FrontendFillConfigWithDefaults(FrontendConfig *){}
inline int FrontendPopulateState(const FrontendConfig *,FrontendState *,int){return 1;}
inline void FrontendFreeStateContents(FrontendState *){}
inline FrontendOutput FrontendProcessSamples(FrontendState *state,const int16_t *pcm,size_t size,size_t *consumed){
    *consumed=std::min(size,state->next-state->pcm.size());
    state->pcm.insert(state->pcm.end(),pcm,pcm+*consumed);
    if(state->pcm.size()!=state->next)return {nullptr,0};
    auto window=state->pcm.data()+state->next-480;
    for(unsigned i=0;i<40;++i)state->features[i]=shim::feature(window,i);
    state->next+=160;
    return {state->features,40};
}
}
