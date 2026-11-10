// Experimental streaming detector; no microphone, playback, networking or NVS.
#include "panel_wake_runtime.h"
#include "wake_config.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"
extern "C" {
#include "frontend.h"
#include "frontend_util.h"
}

extern const uint8_t model_bytes[] asm("_binary_hey_nyx_model_bin_start");
#ifdef PANEL_WAKE_SELFTEST
extern const uint8_t selftest_bytes[] asm("_binary_wake_selftest_bin_start");
#endif
namespace {
constexpr size_t arena_size=64*1024, variable_size=16*1024;
constexpr int chunk_samples=480, smooth_steps=5, refractory_steps=67;
constexpr char tag[]="panel.wake.research";

struct Detector {
    uint8_t *arena=nullptr, *variable_arena=nullptr;
    tflite::MicroMutableOpResolver<14> ops;
    tflite::MicroInterpreter *interpreter=nullptr;
    TfLiteTensor *input=nullptr, *output=nullptr;
    FrontendState frontend{};
    bool frontend_started=false, failed=false, armed=true;
    unsigned feature_count=0, smooth_count=0, smooth_at=0;
    int64_t steps=0, last=-1000;
    float history[smooth_steps]{};
    uint64_t inference_us=0;
    unsigned inference_max_us=0;
    unsigned pcm_peak=0;
    double maximum_average=0;
    uint8_t last_output=0;

    ~Detector() {
        delete interpreter;
        if(frontend_started)FrontendFreeStateContents(&frontend);
        free(variable_arena);free(arena);
    }

    bool initialize() {
        arena=static_cast<uint8_t *>(heap_caps_aligned_alloc(16,arena_size,MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT));
        variable_arena=static_cast<uint8_t *>(heap_caps_aligned_alloc(16,variable_size,MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT));
        if(!arena || !variable_arena)return false;
        auto *allocator=tflite::MicroAllocator::Create(variable_arena,variable_size);
        if(!allocator)return false;
        auto *variables=tflite::MicroResourceVariables::Create(allocator,20);
        if(!variables)return false;
        if(ops.AddCallOnce()!=kTfLiteOk || ops.AddVarHandle()!=kTfLiteOk ||
           ops.AddReadVariable()!=kTfLiteOk || ops.AddReshape()!=kTfLiteOk ||
           ops.AddConcatenation()!=kTfLiteOk || ops.AddStridedSlice()!=kTfLiteOk ||
           ops.AddAssignVariable()!=kTfLiteOk || ops.AddConv2D()!=kTfLiteOk ||
           ops.AddDepthwiseConv2D()!=kTfLiteOk || ops.AddSplitV()!=kTfLiteOk ||
           ops.AddFullyConnected()!=kTfLiteOk || ops.AddLogistic()!=kTfLiteOk ||
           ops.AddQuantize()!=kTfLiteOk || ops.AddExpandDims()!=kTfLiteOk)return false;
        auto *model=tflite::GetModel(model_bytes);
        if(model->version()!=TFLITE_SCHEMA_VERSION)return false;
        interpreter=new(std::nothrow)tflite::MicroInterpreter(model,ops,arena,arena_size,variables);
        if(!interpreter || interpreter->AllocateTensors()!=kTfLiteOk)return false;
        input=interpreter->input(0);output=interpreter->output(0);
        if(!input || !output || input->type!=kTfLiteInt8 || input->bytes!=120 ||
           !std::isfinite(input->params.scale) || input->params.scale<=0 ||
           output->type!=kTfLiteUInt8 || output->bytes!=1 ||
           !std::isfinite(output->params.scale) || output->params.scale<=0)return false;
        FrontendConfig config{};
        FrontendFillConfigWithDefaults(&config);
        config.window.size_ms=30;config.window.step_size_ms=10;
        config.filterbank.num_channels=40;
        config.filterbank.lower_band_limit=125;config.filterbank.upper_band_limit=7500;
        config.noise_reduction.smoothing_bits=10;
        config.noise_reduction.even_smoothing=0.025;
        config.noise_reduction.odd_smoothing=0.06;
        config.noise_reduction.min_signal_remaining=0.05;
        config.pcan_gain_control.enable_pcan=1;config.pcan_gain_control.strength=0.95;
        config.pcan_gain_control.offset=80;config.pcan_gain_control.gain_bits=21;
        config.log_scale.enable_log=1;config.log_scale.scale_shift=6;
        frontend_started=true; // Populate can fail after partial allocation.
        if(!FrontendPopulateState(&config,&frontend,16000))return false;
        ESP_LOGI(tag,"model=%s threshold=%.6f tensor=%u allocated=%u+%u",
            PANEL_WAKE_MODEL_SHA256,double(PANEL_WAKE_THRESHOLD),
            unsigned(interpreter->arena_used_bytes()),unsigned(arena_size),unsigned(variable_size));
        return true;
    }

    int detect(const int16_t *pcm) {
        if(failed)return -1;
        for(int i=0;i<chunk_samples;++i)
            pcm_peak=std::max(pcm_peak,unsigned(std::abs(int(pcm[i]))));
        bool triggered=false;
        size_t at=0;
        while(at<chunk_samples) {
            size_t consumed=0;
            auto features=FrontendProcessSamples(&frontend,pcm+at,std::min(size_t(160),size_t(chunk_samples)-at),&consumed);
            if(!consumed || consumed>size_t(chunk_samples)-at){failed=true;return -1;}
            at+=consumed;
            if(!features.size)continue;
            if(features.size!=40 || feature_count>=3){failed=true;return -1;}
            for(unsigned i=0;i<40;++i) {
                float value=std::clamp(features.values[i]/25.6f,0.f,26.f);
                input->data.int8[feature_count*40+i]=std::clamp(
                    int(std::nearbyint(value/input->params.scale))+int(input->params.zero_point),-128,127);
            }
            if(++feature_count!=3)continue;
            feature_count=0;
            int64_t started=esp_timer_get_time();
            if(interpreter->Invoke()!=kTfLiteOk){failed=true;ESP_LOGE(tag,"Inference failed");return -1;}
            last_output=output->data.uint8[0];
            unsigned elapsed=unsigned(esp_timer_get_time()-started);
            inference_us+=elapsed;inference_max_us=std::max(inference_max_us,elapsed);
            float score=(int(output->data.uint8[0])-int(output->params.zero_point))*output->params.scale;
            if(!std::isfinite(score)){failed=true;return -1;}
            ++steps;
            // Recompute the five-value sum in time order, like the host's
            // convolution. A rolling subtract/add would accumulate drift.
            history[smooth_at]=score;smooth_at=(smooth_at+1)%smooth_steps;
            if(smooth_count<smooth_steps)++smooth_count;
            if(smooth_count!=smooth_steps)continue;
            double total=0;
            for(unsigned k=0;k<smooth_steps;++k)total+=history[(smooth_at+k)%smooth_steps];
            double average=total/smooth_steps;
            maximum_average=std::max(maximum_average,average);
            if(steps%167==0) {
                ESP_LOGI(tag,"progress step=%lld peak=%u max_average=%.6f mean_us=%u max_us=%u stack_free=%u",
                    (long long)steps,pcm_peak,maximum_average,
                    unsigned(inference_us/uint64_t(steps)),inference_max_us,
                    unsigned(uxTaskGetStackHighWaterMark(nullptr)));
                pcm_peak=0;maximum_average=0;
            }
            if(average<PANEL_WAKE_THRESHOLD*.5 && steps-last>=refractory_steps)armed=true;
            if(armed && average>=PANEL_WAKE_THRESHOLD && steps-last>=refractory_steps) {
                triggered=true;armed=false;last=steps;
                ESP_LOGI(tag,"activation average=%.6f step=%lld mean_us=%u max_us=%u",
                    average,(long long)steps,unsigned(inference_us/uint64_t(steps)),inference_max_us);
            }
        }
        return triggered?1:0;
    }
};

model_iface_data_t *create(const void *,det_mode_t) {
#ifdef PANEL_WAKE_SELFTEST
    static bool checked=false;
    if(!checked) {
        auto *probe=new(std::nothrow)Detector;
        if(!probe)return nullptr;
        bool ok=probe->initialize();
        uint32_t samples=0,outputs=0;
        memcpy(&samples,selftest_bytes+40,4);memcpy(&outputs,selftest_bytes+44,4);
        const auto *pcm=reinterpret_cast<const int16_t *>(selftest_bytes+48);
        const auto *expected=selftest_bytes+48+samples*2;
        unsigned mismatches=0;
        if(ok) {
            for(uint32_t chunk=0;chunk<samples/chunk_samples;++chunk) {
                if(probe->detect(pcm+chunk*chunk_samples)<0){ok=false;break;}
                if(chunk && (probe->steps!=chunk || probe->last_output!=expected[chunk-1])) {
                    if(mismatches<5)ESP_LOGE(tag,"selftest step=%u expected=%u actual=%u",
                        unsigned(chunk),unsigned(expected[chunk-1]),unsigned(probe->last_output));
                    ++mismatches;
                }
            }
            ok=ok && probe->steps==outputs && !mismatches;
        }
        ESP_LOGI(tag,"exact-PCM selftest outputs=%u mismatches=%u passed=%s",
            unsigned(outputs),mismatches,ok?"true":"false");
        delete probe; // Never mix diagnostic history with the microphone.
        if(!ok)return nullptr;
        checked=true;
    }
#endif
    auto *detector=new(std::nothrow)Detector;
    if(!detector)return nullptr;
    if(!detector->initialize()){delete detector;return nullptr;}
    return reinterpret_cast<model_iface_data_t *>(detector);
}
void destroy(model_iface_data_t *model) {delete reinterpret_cast<Detector *>(model);}
wakenet_state_t detect(model_iface_data_t *model,int16_t *pcm) {
    return static_cast<wakenet_state_t>(reinterpret_cast<Detector *>(model)->detect(pcm));
}
int chunk(model_iface_data_t *) {return chunk_samples;}
int rate(model_iface_data_t *) {return 16000;}
int channels(model_iface_data_t *) {return 1;}
}

extern "C" const esp_wn_iface_t panel_wake_runtime_iface={
    .create=create,.get_start_point=nullptr,.get_samp_chunksize=chunk,
    .get_channel_num=channels,.get_samp_rate=rate,.get_word_num=nullptr,
    .get_word_name=nullptr,.set_det_threshold=nullptr,.reset_det_threshold=nullptr,
    .get_det_threshold=nullptr,.get_triggered_channel=nullptr,.get_vol_gain=nullptr,
    .detect=detect,.detect_mfcc=nullptr,.get_mfcc_data=nullptr,.clean=nullptr,.destroy=destroy
};
extern "C" int panel_wake_runtime_prime_samples(void) {return PANEL_WAKE_PRIME_SAMPLES;}
