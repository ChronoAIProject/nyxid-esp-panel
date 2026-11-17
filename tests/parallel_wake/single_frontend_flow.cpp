// Exercise the real default C161 adapter through its public create/destroy API.
// Constructed in-memory PCM only; no microphone, playback or recognition claim.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include "panel_wake_runtime.h"
#include "wake_config.h"
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

namespace {
std::map<void*, size_t> single_arenas, single_frontend;
int arena_calls=0, arena_fail_at=-1;
int frontend_calls=0, frontend_fail_at=-1;
size_t arena_bytes=0, frontend_bytes=0;
}

void* heap_caps_aligned_alloc(size_t alignment,size_t size,unsigned) {
    if(arena_calls++==arena_fail_at)return nullptr;
    void* result=nullptr;
    if(posix_memalign(&result,alignment,size))return nullptr;
    single_arenas.emplace(result,size);arena_bytes+=size;
    return result;
}
void* heap_caps_malloc(size_t size,unsigned caps) {
    return heap_caps_aligned_alloc(16,size,caps);
}
void nyx_single_test_free(void* data) {
    if(!data)return;
    assert(single_arenas.count(data)==1);
    arena_bytes-=single_arenas.at(data);single_arenas.erase(data);std::free(data);
}
void heap_caps_free(void* data) { nyx_single_test_free(data); }
extern "C" void* nyx_test_frontend_malloc(size_t size) {
    if(frontend_calls++==frontend_fail_at)return nullptr;
    void* result=std::malloc(size);assert(result);
    single_frontend.emplace(result,size);frontend_bytes+=size;return result;
}
extern "C" void* nyx_test_frontend_calloc(size_t count,size_t size) {
    void* result=nyx_test_frontend_malloc(count*size);
    if(result)std::memset(result,0,count*size);
    return result;
}
extern "C" void nyx_test_frontend_free(void* data) {
    if(!data)return;
    assert(single_frontend.count(data)==1);
    frontend_bytes-=single_frontend.at(data);single_frontend.erase(data);std::free(data);
}

// Intercept only the real adapter's two explicit arena frees. Dependencies and
// the C++ runtime keep their normal allocator behavior; frontend C TUs are
// intercepted separately by the runner.
#define free nyx_single_test_free
#include "panel_wake_runtime.cpp"
#undef free
#include "single_model.h"

void assert_single_released() {
    assert(single_arenas.empty() && arena_bytes==0);
    assert(single_frontend.empty() && frontend_bytes==0);
}

void assert_fresh_public_create() {
    auto* model=panel_wake_runtime_iface.create(nullptr,DET_MODE_90);assert(model);
    int16_t silence[480]{};
    assert(panel_wake_runtime_iface.detect(model,silence)>=0);
    panel_wake_runtime_iface.destroy(model);assert_single_released();
}

int main() {
    auto* model=panel_wake_runtime_iface.create(nullptr,DET_MODE_90);assert(model);
    int arena_sites=arena_calls, frontend_sites=frontend_calls;
    const size_t reserved_arena=arena_bytes, reserved_frontend=frontend_bytes;
    assert(arena_sites==2 && frontend_sites==17);
    assert(panel_wake_runtime_iface.get_samp_chunksize(model)==480);
    assert(panel_wake_runtime_iface.get_samp_rate(model)==16000);
    assert(panel_wake_runtime_iface.get_channel_num(model)==1);
    const int prime=panel_wake_runtime_prime_samples();
    const int blocks=(prime+479)/480;assert(blocks*480==81600);
    auto* detector=reinterpret_cast<Detector*>(model);
    int16_t pcm[480]{};
    for(int block=0;block<blocks;++block)assert(panel_wake_runtime_iface.detect(model,pcm)>=0);
    assert(detector->steps==169 && detector->feature_count==1);
    const auto* window=detector->frontend.window.input;
    const auto* interpreter=detector->interpreter;
    for(int block=0;block<12;++block) {
        for(int i=0;i<480;++i)pcm[i]=int16_t(((i+480*block)*97)%8192-4096);
        assert(panel_wake_runtime_iface.detect(model,pcm)>=0);
        assert(detector->steps==170+block && detector->feature_count==1);
        assert(detector->frontend.window.input==window && detector->interpreter==interpreter);
    }
    panel_wake_runtime_iface.destroy(model);assert_single_released();
    for(int failure=0;failure<arena_sites;++failure) {
        arena_calls=0;arena_fail_at=failure;
        assert(!panel_wake_runtime_iface.create(nullptr,DET_MODE_90));assert_single_released();
        arena_fail_at=-1;assert_fresh_public_create();
    }
    for(int failure=0;failure<frontend_sites;++failure) {
        frontend_calls=0;frontend_fail_at=failure;
        std::fprintf(stderr,"SINGLE_FRONTEND_ALLOCATION_FAULT %d\n",failure);
        assert(!panel_wake_runtime_iface.create(nullptr,DET_MODE_90));assert_single_released();
        frontend_fail_at=-1;assert_fresh_public_create();
    }
    std::printf("HOST_SINGLE_C161_FLOW_OK prime_samples=81600 prime_calls=169 live_calls=12 arena_fault_sites=%d frontend_fault_sites=%d arena_reserved=%zu frontend_reserved=%zu\n",
                arena_sites,frontend_sites,reserved_arena,reserved_frontend);
}
