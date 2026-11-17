// Actual TFLM model/allocator smoke. No PCM, audio, frontend or score reference.
#include "parallel_model.h"
#include <array>
#include <cassert>
#include <fstream>
#include <iterator>
#include <map>
#include <vector>

namespace {
std::map<void*, size_t> allocations;
int allocation_calls = 0, fail_at = -1;
size_t allocated_bytes = 0, peak_bytes = 0;

std::vector<uint8_t> read(const char* path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), {}};
}
}

void* heap_caps_aligned_alloc(size_t alignment, size_t size, unsigned) {
    if (allocation_calls++ == fail_at) return nullptr;
    void* result = nullptr;
    if (posix_memalign(&result, alignment, size)) return nullptr;
    allocations.emplace(result, size);
    allocated_bytes += size;
    peak_bytes = std::max(peak_bytes, allocated_bytes);
    return result;
}
void* heap_caps_malloc(size_t size, unsigned caps) {
    return heap_caps_aligned_alloc(16, size, caps);
}
void heap_caps_free(void* data) {
    if (!data) return;
    assert(allocations.count(data) == 1);
    allocated_bytes -= allocations.at(data);
    allocations.erase(data);
    std::free(data);
}

int main(int argc, char** argv) {
    assert(argc == 4);
    std::vector<uint8_t> model_bytes[] = {read(argv[1]), read(argv[2]), read(argv[3])};
    const char* names[] = {"c161", "binary40", "history_expert"};
    int total_allocations = 0;
    {
        panel_wake_parallel::Model models[3];
        for (int i=0; i<3; ++i) {
            assert(!model_bytes[i].empty());
            assert(models[i].initialize(model_bytes[i].data(), names[i], i != 0));
        }
        total_allocations = allocation_calls;
        std::printf("HOST_RESIDENT_WRAPPER_BYTES %zu\n", allocated_bytes);
        // Actual signatures map cache6 to input0. This tests name resolution,
        // not a duplicate positional cache implementation in the test.
        assert(models[1].cache_inputs[6] == models[1].interpreter->input(0));
        for (int i=0;i<7;++i) {
            assert(models[1].retained[i].data != models[1].cache_outputs[i]->data.int8);
            assert(models[1].retained[i].data != models[2].retained[i].data);
        }
        // Two independent interpreter sets receive the same constructed
        // features with intervening use of the other models. Equal fresh
        // streams establish cache ownership/reset determinism on this probe,
        // not P4 numerical equality or recognition acceptance.
        panel_wake_parallel::Model fresh[3];
        for(int i=0;i<3;++i) assert(fresh[i].initialize(model_bytes[i].data(), names[i], i!=0));
        for(int step=0;step<24;++step) {
            float features[120];
            for(int i=0;i<120;++i) features[i]=float((i*17+step*31)%261)/10.f;
            for(int i=0;i<3;++i) {
                uint8_t a=0,b=0;
                assert(models[i].invoke(features,a));
                assert(fresh[i].invoke(features,b));
                assert(a==b);
            }
        }
    }
    assert(allocated_bytes==0 && allocations.empty());
    // Fail every wrapper allocation across all models, retaining ordinary
    // destruction. No failed initialization may leak or double-free storage.
    for(int failure=0;failure<total_allocations;++failure) {
        allocation_calls=0; fail_at=failure;
        {
            panel_wake_parallel::Model models[3];
            bool initialized=true;
            for(int i=0;i<3 && initialized;++i)
                initialized=models[i].initialize(model_bytes[i].data(),names[i],i!=0);
            assert(!initialized);
        }
        assert(allocated_bytes==0 && allocations.empty());
    }
    std::printf("HOST_FLOW_OK model_invocations=144 fault_sites=%d peak_two_sets=%zu\n",total_allocations,peak_bytes);
}
