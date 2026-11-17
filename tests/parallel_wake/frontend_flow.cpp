// Whole shared frontend/model chain with in-memory constructed PCM only.
// No device access, playback, recording, speech baseline or recognition claim.
#include "parallel_detector.h"
#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <vector>

namespace {
std::map<void*, size_t> allocations;
std::map<void*, size_t> frontend_allocations;
int allocation_calls = 0, fail_at = -1;
int frontend_allocation_calls = 0, frontend_fail_at = -1;
size_t allocated_bytes = 0, frontend_allocated_bytes = 0;
std::vector<uint8_t> read(const char* path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), {}};
}
}
void* heap_caps_aligned_alloc(size_t alignment, size_t size, unsigned) {
    if (allocation_calls++ == fail_at) return nullptr;
    void* result=nullptr;
    if (posix_memalign(&result,alignment,size)) return nullptr;
    allocations.emplace(result,size); allocated_bytes+=size;
    return result;
}
void* heap_caps_malloc(size_t size, unsigned caps) { return heap_caps_aligned_alloc(16,size,caps); }
void heap_caps_free(void* data) {
    if (!data) return;
    assert(allocations.count(data)==1);
    allocated_bytes-=allocations.at(data); allocations.erase(data); std::free(data);
}
extern "C" void* nyx_test_frontend_malloc(size_t size) {
    if (frontend_allocation_calls++ == frontend_fail_at) return nullptr;
    void* result=std::malloc(size);
    assert(result);
    frontend_allocations.emplace(result,size); frontend_allocated_bytes+=size;
    return result;
}
extern "C" void* nyx_test_frontend_calloc(size_t count,size_t size) {
    void* result=nyx_test_frontend_malloc(count*size);
    if (result) std::memset(result,0,count*size);
    return result;
}
extern "C" void nyx_test_frontend_free(void* data) {
    if (!data) return;
    assert(frontend_allocations.count(data)==1);
    frontend_allocated_bytes-=frontend_allocations.at(data);
    frontend_allocations.erase(data); std::free(data);
}
void assert_released() {
    assert(allocated_bytes==0 && allocations.empty());
    assert(frontend_allocated_bytes==0 && frontend_allocations.empty());
}

int main(int argc,char** argv) {
    assert(argc==4);
    const std::vector<uint8_t> model[] = {read(argv[1]),read(argv[2]),read(argv[3])};
    for(const auto& data:model) assert(!data.empty());
    int total_allocations=0, total_frontend_allocations=0;
    {
        panel_wake_parallel::ParallelDetector detector;
        assert(detector.initialize(model[0].data(),model[1].data(),model[2].data()));
        total_allocations=allocation_calls;
        total_frontend_allocations=frontend_allocation_calls;
        int16_t pcm[480]{};
        // The original 30ms window, 10ms hop and 3-frame stride imply
        // floor((N/160-2)/3) calls. Prime is 170 native detector blocks.
        for(int block=0;block<170;++block) assert(detector.detect(pcm)>=0);
        assert(detector.steps==169 && detector.feature_count==1);
        const auto* frontend_window=detector.frontend.window.input;
        auto* c161_interpreter=detector.models[0].interpreter;
        auto* retained=detector.models[1].retained[0].data;
        // At the transition, the existing pending feature must become the
        // first frame of the next model chunk, rather than being reset.
        for(int block=0;block<12;++block) {
            for(int i=0;i<480;++i) pcm[i]=int16_t(((i+480*block)*97)%8192-4096);
            const uint64_t before=detector.steps;
            assert(detector.detect(pcm)>=0);
            assert(detector.steps==before+1 && detector.feature_count==1);
            assert(detector.frontend.window.input==frontend_window);
            assert(detector.models[0].interpreter==c161_interpreter);
            assert(detector.models[1].retained[0].data==retained);
        }
        assert(detector.steps==181);
        // Corrupt one explicit-cache ABI after a successful live segment.
        // A partial invocation must not update the shared event step or be
        // reused on future calls after a permanent detector failure.
        detector.models[1].retained[0].scale*=2;
        const uint64_t before=detector.steps;
        assert(detector.detect(pcm)<0 && detector.failed && detector.steps==before);
        assert(detector.detect(pcm)<0 && detector.steps==before);
    }
    assert_released();
    // A failed create/rearm can be followed by a clean fresh detector.
    for(int failure=0;failure<total_allocations;++failure) {
        allocation_calls=0;fail_at=failure;
        {
            panel_wake_parallel::ParallelDetector detector;
            assert(!detector.initialize(model[0].data(),model[1].data(),model[2].data()));
        }
        assert_released();
    }
    fail_at=-1;
    for(int failure=0;failure<total_frontend_allocations;++failure) {
        frontend_allocation_calls=0;frontend_fail_at=failure;
        std::fprintf(stderr,"FRONTEND_ALLOCATION_FAULT %d\n",failure);
        {
            panel_wake_parallel::ParallelDetector detector;
            assert(!detector.initialize(model[0].data(),model[1].data(),model[2].data()));
        }
        assert_released();
    }
    frontend_fail_at=-1;
    {
        panel_wake_parallel::ParallelDetector fresh;
        assert(fresh.initialize(model[0].data(),model[1].data(),model[2].data()));
        int16_t silence[480]{};
        assert(fresh.detect(silence)>=0 && fresh.steps==0 && fresh.feature_count==1);
    }
    assert_released();
    std::printf("HOST_FRONTEND_FLOW_OK prime_samples=81600 prime_calls=169 live_calls=12 wrapper_fault_sites=%d frontend_fault_sites=%d\n",total_allocations,total_frontend_allocations);
}
