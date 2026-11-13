// Includes the real runtime; all external dependencies are host test doubles.
#include "panel_wake_runtime.cpp"
#include <cassert>
#include <iostream>

extern const uint8_t model_bytes[]={0};

std::vector<int16_t> source(size_t count) {
    std::vector<int16_t> pcm(count);
    for(size_t i=0;i<count;++i)pcm[i]=int16_t((i*53+7)%60001-30000);
    return pcm;
}

std::vector<shim::Group> oracle(std::vector<int16_t> pcm,bool complete) {
    if(complete)pcm.resize((pcm.size()+159)/160*160,0);
    std::vector<std::array<int8_t,40>> frames;
    for(size_t end=480;end<=pcm.size();end+=160) {
        std::array<int8_t,40> frame;
        // Features are integer multiples of32, giving exact1.25-unit values
        // and integer5-unit input steps in this constructed frontend.
        for(unsigned channel=0;channel<40;++channel)
            frame[channel]=int8_t((shim::feature(pcm.data()+end-480,channel)/32)*5-90);
        frames.push_back(frame);
    }
    if(complete)while(frames.size()%3){std::array<int8_t,40> zero;zero.fill(-90);frames.push_back(zero);}
    std::vector<shim::Group> result;
    for(size_t i=0;i+2<frames.size();i+=3) {
        shim::Group group;
        for(unsigned j=0;j<3;++j)std::copy(frames[i+j].begin(),frames[i+j].end(),group.begin()+j*40);
        result.push_back(group);
    }
    return result;
}

void check_events(const Detector &detector) {
    bool armed=true;int64_t last=-1000;
    for(size_t i=4;i<shim::groups.size();++i) {
        unsigned sum=0;
        for(size_t j=i-4;j<=i;++j)sum+=shim::raw(shim::groups[j]);
        double average=double(sum)/1280;
        auto step=int64_t(i+1);
        if(average<double(PANEL_WAKE_THRESHOLD)*.5 && step-last>=67)armed=true;
        if(armed && average>=double(PANEL_WAKE_THRESHOLD) && step-last>=67){last=step;armed=false;}
    }
    assert(detector.last==last && detector.armed==armed);
    assert(detector.steps==int64_t(shim::groups.size()));
}

#ifndef FLOW_LEGACY_ORIGINAL
void stream_case(size_t samples,const std::vector<size_t> &chunks) {
    shim::reset();Detector detector;assert(detector.initialize());
    const auto pcm=source(samples);
    size_t at=0,index=0;
    while(at<samples) {
        const size_t take=std::min(chunks[index++%chunks.size()],samples-at);
        assert(detector.detect(pcm.data()+at,take)>=0);at+=take;
        check_events(detector);
    }
    assert(detector.frontend.pcm==pcm);
    assert(shim::groups==oracle(pcm,false));
#ifdef PANEL_WAKE_COMPLETE_INPUT
    auto full=oracle(pcm,true);
    std::vector<uint8_t> raw;
    for(const auto &group:full)raw.push_back(shim::raw(group));
    detector.expected_trace=raw.data();detector.expected_outputs=uint32_t(raw.size());
    assert(!detector.finish(samples+1));
    assert(detector.finish(samples));
    assert(shim::groups==full);
    auto padded=pcm;padded.resize((samples+159)/160*160,0);
    assert(detector.frontend.pcm==padded);
    assert(!detector.trace_mismatches);
    check_events(detector);
    auto before=shim::groups;
    assert(!detector.finish(samples));
    assert(detector.detect(pcm.data(),1)==-1);
    assert(shim::groups==before);
#endif
}
#endif

int main() {
#ifdef FLOW_SELFTEST_CASE
    shim::reset();
    auto *created=create(nullptr,0);
    assert(bool(created)==bool(FLOW_EXPECT_SELFTEST_PASS));
    if(created){destroy(created);auto *second=create(nullptr,0);assert(second);destroy(second);}
    std::cout<<"{\"selftest_case_passed\":true}\n";
#else
    size_t cases=0;
#ifndef FLOW_LEGACY_ORIGINAL
    for(size_t samples:std::vector<size_t>{960,961,1119,1120,1279,1280,1281,50000,68000})
        for(const auto &chunks:std::vector<std::vector<size_t>>{{480},{1,159,5,1024},{samples}}){
            stream_case(samples,chunks);++cases;
        }
    shim::reset();shim::fail_at=0;
    Detector failed;assert(failed.initialize());auto pcm=source(960);
    assert(failed.detect(pcm.data(),pcm.size())==-1);
    assert(failed.detect(pcm.data(),1)==-1);
#ifdef PANEL_WAKE_COMPLETE_INPUT
    assert(!failed.finish(960));
    for(unsigned mode=0;mode<2;++mode){
        shim::reset();shim::bad_shape=mode==0;shim::bad_quant=mode==1;
        Detector invalid;assert(!invalid.initialize());
    }
    shim::reset();Detector mismatch;assert(mismatch.initialize());
    const uint8_t wrong=255;mismatch.expected_trace=&wrong;mismatch.expected_outputs=1;
    assert(mismatch.detect(pcm.data(),pcm.size())>=0);
    assert(mismatch.trace_mismatches==1);
#endif
#endif
    std::cout<<"{\"stream_cases\":"<<cases<<",\"passed\":true,\"groups\":[";
    // A deterministic continuous prefix for cross-source regression checks.
    shim::reset();Detector prefix;assert(prefix.initialize());auto long_pcm=source(100320);
    for(size_t at=0;at<long_pcm.size();at+=480)assert(prefix.detect(long_pcm.data()+at)>=0);
    for(size_t i=0;i<shim::groups.size();++i){if(i)std::cout<<',';std::cout<<unsigned(shim::raw(shim::groups[i]));}
    std::cout<<"],\"last\":"<<prefix.last<<",\"armed\":"<<prefix.armed<<"}\n";
#endif
}
