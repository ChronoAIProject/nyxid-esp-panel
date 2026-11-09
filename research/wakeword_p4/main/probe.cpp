// Numeric probe only: no I2S, networking, credentials, NVS writes or wake decisions.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "esp_heap_caps.h"
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
#include "vectors.h"

extern const uint8_t nabu_model[] asm("_binary_okay_nabu_tflite_start");
extern const uint8_t smoke_model[] asm("_binary_smoke_tflite_start");
static uint16_t board_frontend_features[198*40];

static int8_t quantize_feature(unsigned raw,float scale,int zero) {
    float value=std::clamp(raw/25.6f,0.f,26.f);
    return std::clamp(int(std::nearbyint(value/scale))+zero,-128,127);
}

static bool frontend_probe() {
    FrontendConfig config{};
    FrontendFillConfigWithDefaults(&config);
    config.window.size_ms=30; config.window.step_size_ms=10;
    config.filterbank.num_channels=40;
    config.filterbank.lower_band_limit=125; config.filterbank.upper_band_limit=7500;
    config.noise_reduction.smoothing_bits=10;
    config.noise_reduction.even_smoothing=0.025;
    config.noise_reduction.odd_smoothing=0.06;
    config.noise_reduction.min_signal_remaining=0.05;
    config.pcan_gain_control.enable_pcan=1; config.pcan_gain_control.strength=0.95;
    config.pcan_gain_control.offset=80; config.pcan_gain_control.gain_bits=21;
    config.log_scale.enable_log=1; config.log_scale.scale_shift=6;
    FrontendState state{};
    if(!FrontendPopulateState(&config,&state,16000))return false;
    unsigned frames=0,mismatches=0,max_delta=0;
    int64_t total_us=0;
    for(size_t at=0;at<sizeof(test_pcm)/sizeof(test_pcm[0]);) {
        size_t consumed=0;
        int64_t start=esp_timer_get_time();
        auto result=FrontendProcessSamples(&state,test_pcm+at,160,&consumed);
        total_us+=esp_timer_get_time()-start;
        if(!consumed){FrontendFreeStateContents(&state);return false;}
        at+=consumed;
        if(result.size) {
            if(result.size!=40 || frames>=198){FrontendFreeStateContents(&state);return false;}
            for(size_t i=0;i<40;++i) {
                unsigned expected=expected_features[frames*40+i];
                unsigned delta=abs(int(result.values[i])-int(expected));
                if(delta) {
                    if(mismatches<8)printf("PROBE_FRONTEND_DIFFERENCE {\"frame\":%u,\"channel\":%u,\"actual\":%u,\"expected\":%u}\n",
                        frames,unsigned(i),unsigned(result.values[i]),expected);
                    ++mismatches;
                }
                board_frontend_features[frames*40+i]=result.values[i];
                max_delta=std::max(max_delta,delta);
            }
            ++frames;
        }
    }
    FrontendFreeStateContents(&state);
    printf("PROBE_FRONTEND {\"frames\":%u,\"mismatches\":%u,\"max_delta\":%u,\"bit_exact\":%s,\"total_us\":%lld}\n",
           frames,mismatches,max_delta,
           mismatches==0?"true":"false",(long long)total_us);
    // Report cross-platform rounding explicitly. Also test these board-produced
    // features through each model below; this is bounded numeric compatibility,
    // not bit-exact equivalence or wake-word accuracy validation.
    return frames==198 && max_delta<=1;
}

static bool model_probe(const char *name,const uint8_t *data,const int8_t *test_frames,const uint8_t *expected,bool board_frontend=false) {
    constexpr size_t arena_size=64*1024,var_size=16*1024;
    auto *arena=(uint8_t *)heap_caps_aligned_alloc(16,arena_size,MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT);
    auto *var_arena=(uint8_t *)heap_caps_aligned_alloc(16,var_size,MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT);
    if(!arena || !var_arena){free(arena);free(var_arena);return false;}
    bool passed=false;
    {
        auto *allocator=tflite::MicroAllocator::Create(var_arena,var_size);
        auto *variables=tflite::MicroResourceVariables::Create(allocator,20);
        tflite::MicroMutableOpResolver<14> ops;
        ops.AddCallOnce();ops.AddVarHandle();ops.AddReadVariable();ops.AddReshape();
        ops.AddConcatenation();ops.AddStridedSlice();ops.AddAssignVariable();ops.AddConv2D();
        ops.AddDepthwiseConv2D();ops.AddSplitV();ops.AddFullyConnected();ops.AddLogistic();
        ops.AddQuantize();ops.AddExpandDims();
        tflite::MicroInterpreter interpreter(tflite::GetModel(data),ops,arena,arena_size,variables);
        if(interpreter.AllocateTensors()==kTfLiteOk) {
            auto *input=interpreter.input(0);auto *output=interpreter.output(0);
            if(input->type==kTfLiteInt8 && input->bytes==120 && input->params.scale>0 && output->type==kTfLiteUInt8) {
                std::vector<int64_t> times(test_step_count);
                unsigned mismatches=0,max_delta=0,quantized_mismatches=0,quantized_max_delta=0;
                passed=true;
                for(size_t i=0;i<test_step_count;++i) {
                    memcpy(input->data.int8,test_frames+120*i,120);
                    if(board_frontend && i<66) {
                        for(unsigned k=0;k<120;++k) {
                            int8_t actual=quantize_feature(board_frontend_features[120*i+k],input->params.scale,input->params.zero_point);
                            unsigned delta=abs(int(actual)-int(test_frames[120*i+k]));
                            if(delta)++quantized_mismatches;
                            quantized_max_delta=std::max(quantized_max_delta,delta);
                            input->data.int8[k]=actual;
                        }
                    }
                    int64_t start=esp_timer_get_time();
                    if(interpreter.Invoke()!=kTfLiteOk){passed=false;break;}
                    times[i]=esp_timer_get_time()-start;
                    unsigned delta=abs(int(output->data.uint8[0])-int(expected[i]));
                    if(delta)++mismatches;
                    max_delta=std::max(max_delta,delta);
                }
                if(passed) {
                    std::sort(times.begin(),times.end());
                    printf("PROBE_MODEL {\"name\":\"%s\",\"board_frontend\":%s,\"calls\":%u,\"arena_used\":%u,\"var_arena_allocated\":%u,\"p50_us\":%lld,\"p95_us\":%lld,\"max_us\":%lld,\"mismatches\":%u,\"max_output_delta\":%u,\"input_quantized_mismatches\":%u,\"input_quantized_max_delta\":%u,\"input_scale\":%.10f,\"input_zero\":%d}\n",
                        name,board_frontend?"true":"false",test_step_count,
                        unsigned(interpreter.arena_used_bytes()),unsigned(var_size),
                        (long long)times[test_step_count/2],(long long)times[test_step_count*95/100],(long long)times.back(),mismatches,max_delta,
                        quantized_mismatches,quantized_max_delta,double(input->params.scale),int(input->params.zero_point));
                    passed=max_delta<=1 && quantized_max_delta<=1;
                }
            }
        }
    }
    free(arena);free(var_arena);return passed;
}

extern "C" void app_main() {
    bool frontend=frontend_probe();
    bool nabu=model_probe("okay_nabu",nabu_model,test_frames_okay_nabu,expected_okay_nabu);
    bool smoke=model_probe("mac_smoke",smoke_model,test_frames_smoke,expected_smoke);
    bool nabu_chain=model_probe("okay_nabu",nabu_model,test_frames_okay_nabu,expected_okay_nabu,true);
    bool smoke_chain=model_probe("mac_smoke",smoke_model,test_frames_smoke,expected_smoke,true);
    printf("P4_WAKE_PROBE_DONE {\"frontend\":%s,\"pretrained_model\":%s,\"mac_exported_model\":%s,\"frontend_to_pretrained\":%s,\"frontend_to_mac_exported\":%s,\"stack_free\":%u}\n",
           frontend?"true":"false",nabu?"true":"false",smoke?"true":"false",
           nabu_chain?"true":"false",smoke_chain?"true":"false",
           unsigned(uxTaskGetStackHighWaterMark(nullptr)));
}
