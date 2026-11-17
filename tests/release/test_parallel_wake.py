"""Research packaging and explicit-cache isolation, with no acoustic replay."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
COMPONENT = ROOT / 'components/panel_wake_runtime'
spec = importlib.util.spec_from_file_location('parallel_wake', COMPONENT / 'prepare_parallel_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class ParallelWake(unittest.TestCase):
    def test_wrong_research_model_is_rejected_before_packaging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'models').mkdir()
            (root / 'models/c161.tflite').write_bytes(b'wrong model')
            with self.assertRaises(ValueError):
                runtime.prepare(root, root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_cache_ownership_and_signed_probability_paths(self):
        # Distinct storage represents model A retained/input/output and model B
        # retained state. Overwrite all arena views after feedback: only each
        # model's retained bytes may be fed into its next call.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'check.cpp'
            source.write_text(r'''
#include "explicit_cache_bytes_v1.h"
#include <cassert>
#include <limits>
int main() {
    int8_t saved_a[256], input_a[256], output_a[256], saved_b[256];
    nyx::Int8Bytes a{saved_a, 256, .25f, -128};
    nyx::Int8Bytes input{input_a, 256, .25f, -128};
    nyx::Int8Bytes output{output_a, 256, .25f, -128};
    nyx::Int8Bytes b{saved_b, 256, .5f, 17};
    assert(nyx::reset_explicit_cache(a) && nyx::reset_explicit_cache(b));
    for (int i=0; i<256; ++i) { assert(saved_a[i]==-128 && saved_b[i]==17); output_a[i]=i-128; }
    assert(nyx::feedback_explicit_cache(a, output));
    for (auto& code:output_a) code=0;
    assert(nyx::feedback_explicit_cache(input, a));
    for (int i=0; i<256; ++i) { assert(input_a[i]==i-128 && saved_b[i]==17); }
    // Wrong model scale, invalid metadata or length must never mutate A.
    auto invalid=output;
    for (int kind=0; kind<6; ++kind) {
        invalid=output;
        if(kind==0) invalid.scale=.5f;
        if(kind==1) invalid.zero=17;
        if(kind==2) invalid.size=255;
        if(kind==3) invalid.data=nullptr;
        if(kind==4) invalid.scale=std::numeric_limits<float>::quiet_NaN();
        if(kind==5) invalid.zero=128;
        assert(!nyx::feedback_explicit_cache(a, invalid));
        for(int i=0; i<256; ++i) assert(saved_a[i]==i-128);
    }
    int8_t signed_code=0;
    nyx::Int8Bytes probability{&signed_code,1,1.f/256,-128};
    uint8_t decoded=0;
    for(int i=-128;i<128;++i) { signed_code=i; assert(nyx::signed_probability_code(probability,decoded)); assert(decoded==i+128); }
    probability.zero=0; decoded=99;
    assert(!nyx::signed_probability_code(probability,decoded) && decoded==99);
}
''')
            subprocess.run(['c++', '-std=c++11', '-Wall', '-Wextra', '-Werror',
                            '-I' + str(COMPONENT / 'portable'), str(source),
                            '-o', str(root / 'check')], check=True, capture_output=True)
            subprocess.run([str(root / 'check')], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
