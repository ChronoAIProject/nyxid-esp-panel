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
#ifdef PANEL_WAKE_COMPLETE_INPUT
extern const uint8_t selftest_end[] asm("_binary_wake_selftest_bin_end");
#endif
#endif
namespace {
constexpr size_t arena_size=64*1024, variable_size=16*1024;
constexpr int chunk_samples=480, smooth_steps=PANEL_WAKE_SMOOTHING_STEPS, refractory_steps=67;
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
#ifdef PANEL_WAKE_COMPLETE_INPUT
    uint64_t pcm_samples=0;
    bool completed=false;
    const uint8_t *expected_trace=nullptr;
    uint32_t expected_outputs=0, trace_mismatches=0;
#endif

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
#ifdef PANEL_WAKE_COMPLETE_INPUT
        if(!input->dims || input->dims->size!=3 || input->dims->data[0]!=1 ||
           input->dims->data[1]!=3 || input->dims->data[2]!=40 ||
           !output->dims || output->dims->size!=2 || output->dims->data[0]!=1 || output->dims->data[1]!=1 ||
           input->params.zero_point<-128 || input->params.zero_point>127 ||
           output->params.zero_point!=0 || output->params.scale!=1.f/256)return false;
#endif
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
        ESP_LOGI(tag,"model=%s threshold=%.6f smoothing=%d tensor=%u allocated=%u+%u",
            PANEL_WAKE_MODEL_SHA256,double(PANEL_WAKE_THRESHOLD),
            smooth_steps,
            unsigned(interpreter->arena_used_bytes()),unsigned(arena_size),unsigned(variable_size));
        return true;
    }

    bool invoke(bool &triggered) {
        int64_t started=esp_timer_get_time();
        if(interpreter->Invoke()!=kTfLiteOk){failed=true;ESP_LOGE(tag,"Inference failed");return false;}
        last_output=output->data.uint8[0];
        unsigned elapsed=unsigned(esp_timer_get_time()-started);
        inference_us+=elapsed;inference_max_us=std::max(inference_max_us,elapsed);
        float score=(int(output->data.uint8[0])-int(output->params.zero_point))*output->params.scale;
        if(!std::isfinite(score)){failed=true;return false;}
        ++steps;
#ifdef PANEL_WAKE_COMPLETE_INPUT
        if(expected_trace && (uint64_t(steps)>expected_outputs || last_output!=expected_trace[steps-1])) {
            if(trace_mismatches<5)ESP_LOGE(tag,"complete selftest step=%lld actual=%u",(long long)steps,unsigned(last_output));
            ++trace_mismatches;
        }
#endif
        // Keep the original chronological binary64 sum; for the complete
        // profile the output guard above requires exact UINT8 scale1/256.
        history[smooth_at]=score;smooth_at=(smooth_at+1)%smooth_steps;
        if(smooth_count<smooth_steps)++smooth_count;
        if(smooth_count!=smooth_steps)return true;
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
        return true;
    }

    int detect(const int16_t *pcm,size_t samples=chunk_samples) {
        if(failed)return -1;
#ifdef PANEL_WAKE_COMPLETE_INPUT
        if(completed)return -1;
        pcm_samples+=samples;
#endif
        for(size_t i=0;i<samples;++i)
            pcm_peak=std::max(pcm_peak,unsigned(std::abs(int(pcm[i]))));
        bool triggered=false;
        size_t at=0;
        while(at<samples) {
            size_t consumed=0;
            auto features=FrontendProcessSamples(&frontend,pcm+at,std::min(size_t(160),samples-at),&consumed);
            if(!consumed || consumed>samples-at){failed=true;return -1;}
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
            if(!invoke(triggered))return -1;
        }
        return triggered?1:0;
    }

#ifdef PANEL_WAKE_COMPLETE_INPUT
    bool finish(uint64_t original_samples) {
        if(failed || completed || !original_samples || pcm_samples!=original_samples)return false;
        int16_t zeros[160]{};
        size_t final_hop=(160-original_samples%160)%160;
        if(final_hop && detect(zeros,final_hop)<0)return false;
        bool triggered=false;
        if(feature_count) {
            int8_t zero=int8_t(input->params.zero_point);
            while(feature_count<3) {
                std::fill_n(input->data.int8+feature_count*40,40,zero);
                ++feature_count;
            }
            feature_count=0;
            if(!invoke(triggered))return false;
        }
        completed=true;
        return true;
    }
#endif
};

model_iface_data_t *create(const void *,det_mode_t) {
#ifdef PANEL_WAKE_SELFTEST
    static bool checked=false;
    if(!checked) {
        auto *probe=new(std::nothrow)Detector;
        if(!probe)return nullptr;
        bool ok=probe->initialize();
        uint32_t samples=0,outputs=0;
#ifdef PANEL_WAKE_COMPLETE_INPUT
        size_t fixture_size=size_t(selftest_end-selftest_bytes);
        ok=ok && fixture_size>=48;
        if(ok) {
            memcpy(&samples,selftest_bytes+40,4);memcpy(&outputs,selftest_bytes+44,4);
            // Bound sample arithmetic and pointers before using the header.
            ok=!memcmp(selftest_bytes,"NYXKWS02",8) && samples>=960 && samples<=16000*16;
        }
        if(ok) {
            uint32_t frames=(samples+159)/160-2;
            ok=outputs==(frames+2)/3 && fixture_size==48+size_t(samples)*2+outputs;
        }
#else
        memcpy(&samples,selftest_bytes+40,4);memcpy(&outputs,selftest_bytes+44,4);
#endif
        unsigned mismatches=0;
#ifdef PANEL_WAKE_COMPLETE_INPUT
        if(ok) {
            const auto *pcm=reinterpret_cast<const int16_t *>(selftest_bytes+48);
            const auto *expected=selftest_bytes+48+size_t(samples)*2;
            probe->expected_trace=expected;probe->expected_outputs=outputs;
            for(size_t at=0;at<samples;at+=chunk_samples) {
                if(probe->detect(pcm+at,std::min(size_t(chunk_samples),size_t(samples)-at))<0){ok=false;break;}
            }
            ok=ok && probe->finish(samples) && probe->steps==outputs && !probe->trace_mismatches;
            mismatches=probe->trace_mismatches;
        }
#else
        const auto *pcm=reinterpret_cast<const int16_t *>(selftest_bytes+48);
        const auto *expected=selftest_bytes+48+samples*2;
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
#endif
#ifdef PANEL_WAKE_COMPLETE_INPUT
        ESP_LOGI(tag,"complete selftest fixture=%s samples=%u outputs=%u",
            PANEL_WAKE_SELFTEST_SHA256,unsigned(samples),unsigned(outputs));
#endif
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
