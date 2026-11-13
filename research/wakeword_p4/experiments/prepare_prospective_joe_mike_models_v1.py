"""Authenticate two rdh-lineage candidate models; no synthesis or inference."""
import hashlib
import json
from pathlib import Path
import urllib.request
import onnx

PLAN = Path('research/wakeword_p4/results/4090-prospective-joe-mike-source-plan-v1.json')
EXPECTED = 'c8b104364dcb6e5e1a688651226670c582badae681af7c5483e6e2a16cdf1894'

def main():
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == EXPECTED
    plan = json.loads(PLAN.read_text()); rows = []
    for row in plan['models']:
        config = Path(row['config_path'])
        assert hashlib.sha256(config.read_bytes()).hexdigest() == row['config_sha256']
        path = Path(row['model_path']); assert not path.exists()
        partial = path.with_suffix('.onnx.partial')
        h = hashlib.sha256(); size = 0
        print(json.dumps(dict(downloading=row['voice'])), flush=True)
        with urllib.request.urlopen(row['model_url'], timeout=60) as response, partial.open('xb') as output:
            while True:
                block = response.read(4*1024*1024)
                if not block: break
                output.write(block); h.update(block); size += len(block)
        assert h.hexdigest() == row['expected_sha256'] and size == row['expected_size']
        partial.rename(path)
        model = onnx.load(str(path)); onnx.checker.check_model(model)
        describe = lambda v: dict(name=v.name, element_type=v.type.tensor_type.elem_type,
            shape=[d.dim_value if d.HasField('dim_value') else d.dim_param for d in v.type.tensor_type.shape.dim])
        rows.append(dict(voice=row['voice'], path=str(path), size=size, sha256=h.hexdigest(),
            onnx_checker_passed=True, inputs=[describe(v) for v in model.graph.input],
            outputs=[describe(v) for v in model.graph.output],
            random_operators=sorted({n.op_type for n in model.graph.node if 'Random' in n.op_type})))
    report = dict(plan_sha256=EXPECTED, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        models=rows, audio_generated=False, inference_performed=False, roles_reserved=False,
        physical_speaker_independence_proven=False,
        limits='Static downloaded model identity/interface only, not synthesis quality or independent source qualification.')
    output = Path('research/wakeword_p4/results/4090-prospective-joe-mike-model-inventory-v1.json')
    with output.open('x') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report), flush=True)

if __name__ == '__main__': main()
