"""Host C++11 exhaustive-byte checks using both frozen model ABI reports."""
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

R = Path('research/wakeword_p4/results')
header = Path('research/wakeword_p4/portable/explicit_cache_bytes_v1.h')
paths = [R/'4090-parallel-history-abi-original-v1.json',
         R/'4090-binary40-paired-carrier-control-export-original-v1.json']
h, b = [json.loads(p.read_text()) for p in paths]
assert b['model']['sha256'] == '454058039dd41e729ce82ac7f33a84edec4af9554615a632e3b33d47de38009c'
assert h['history_model']['sha256'] == 'd92b95278334043ff70c20c6e8f267e13fe83acac7dbcf705cc2177df9130950'
pairs = h['history_cache_pairs'][:]
for i in range(7):
    a, z = b['inputs'][f'cache{i}'], b['outputs'][f'cache{i}_out']
    assert a == z
    pairs.append(dict(input_bytes=math.prod(a['shape']), input_scale=a['scale'],
                      output_scale=z['scale'], input_real_zero_code=a['zero'], output_real_zero_code=z['zero']))
assert len(pairs) == 14
specs = []
for p in pairs:
    assert p['input_scale'] == p['output_scale'] and p['input_real_zero_code'] == p['output_real_zero_code'] == -128
    specs.append('{%d, %.17g}' % (p['input_bytes'], p['input_scale']))
source = '''#include "explicit_cache_bytes_v1.h"
#include <vector>
#include <limits>
#include <cassert>
int main() {
 struct Spec {std::size_t n; float scale;};
 const Spec specs[] = {SPECS};
 for (const auto& s: specs) {
  std::vector<std::int8_t> a(s.n, 7), b(s.n, 0);
  nyx::Int8Bytes input{a.data(),s.n,s.scale,-128}, output{b.data(),s.n,s.scale,-128};
  assert(nyx::reset_explicit_cache(input));
  for(auto value:a) assert(value == -128);
  for(int code=-128; code<=127; ++code) {
   for(auto& value:b) value=static_cast<std::int8_t>(code);
   assert(nyx::feedback_explicit_cache(input,output));
   assert(a==b);
  }
  auto unchanged=a;
  auto wrong=output; wrong.scale*=2;
  assert(!nyx::feedback_explicit_cache(input,wrong) && a==unchanged);
  wrong=output; wrong.zero=0;
  assert(!nyx::feedback_explicit_cache(input,wrong) && a==unchanged);
  wrong=output; wrong.size++;
  assert(!nyx::feedback_explicit_cache(input,wrong) && a==unchanged);
  wrong=output; wrong.data=nullptr;
  assert(!nyx::feedback_explicit_cache(input,wrong) && a==unchanged);
 }
 for(int value=-128;value<=127;++value) {
  std::int8_t raw=static_cast<std::int8_t>(value); std::uint8_t code=0;
  nyx::Int8Bytes tensor{&raw,1,1.f/256,-128};
  assert(nyx::signed_probability_code(tensor,code) && int(code)==value+128);
  tensor.zero=0; code=77;
  assert(!nyx::signed_probability_code(tensor,code) && code==77);
 }
 std::int8_t raw=7;
 for(float scale: {0.f,-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
  nyx::Int8Bytes invalid{&raw,1,scale,-128};
  assert(!nyx::reset_explicit_cache(invalid) && raw==7);
 }
 return 0;
}
'''.replace('SPECS', ','.join(specs))
with tempfile.TemporaryDirectory(prefix='nyx-cache-bytes-') as folder:
    p = Path(folder); (p/'check.cpp').write_text(source)
    subprocess.run(['c++','-std=c++11','-Wall','-Wextra','-Werror','-pedantic','-I',str(header.parent.resolve()),str(p/'check.cpp'),'-o',str(p/'check')],check=True)
    subprocess.run([str(p/'check')],check=True)
out = dict(all_passed=True, cache_pairs=14, distinct_byte_values_per_pair=256,
           probability_code_values=256, rejected_invalid_metadata_without_mutation=True,
           source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [header,Path(__file__),*paths]},
           generated_cpp_sha256=hashlib.sha256(source.encode()).hexdigest(),
           limits='Host adapter arithmetic only, no TFLM allocation/inference, P4 performance or whole-panel RAM claim.')
with (R/'4090-explicit-cache-bytes-host-check-v1.json').open('x') as f:
    json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
