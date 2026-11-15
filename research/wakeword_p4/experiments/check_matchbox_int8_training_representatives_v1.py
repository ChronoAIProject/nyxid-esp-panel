"""INT8 classification on calibration/training inputs, not a wake metric."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter,OpResolverType
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword/matchbox-window-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    training=R/'4090-matchbox-window-training-original-v1.json';tr=json.loads(training.read_text());export=R/'4090-matchbox-window-int8-original-v1.json';ex=json.loads(export.read_text());assert sha(ex['model']['path'])==ex['model']['sha256']
    for name in ['representative.npy','first-labels.npy']:assert sha(BASE/name)==tr['files'][name]['sha256']
    values=np.load(BASE/'representative.npy',allow_pickle=False);labels=np.load(BASE/'first-labels.npy',allow_pickle=False);assert values.shape==(129,128,1,64) and labels.shape==(128,) and [int((labels==i).sum()) for i in range(7)]==[32,16,16,16,16,16,16]
    it=Interpreter(model_path=ex['model']['path'],num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN);it.allocate_tensors();a=it.get_input_details()[0];b=it.get_output_details()[0];scale,zero=a['quantization'];assert b['quantization']==(1/256,0)
    codes=[]
    for value in values:
        q=np.clip(np.rint(value[None]/scale)+zero,-128,127).astype(np.int8);it.set_tensor(a['index'],q);it.invoke();codes.append(int(it.get_tensor(b['index'])[0,0]))
    groups={}
    for label in range(7):
        group=[codes[i] for i,v in enumerate(labels) if v==label];groups[str(label)]=dict(windows=len(group),minimum_code=min(group),maximum_code=max(group),mean_code=float(np.mean(group)),codes_at_least235=sum(c>=235 for c in group))
    report=dict(schema='matchbox-int8-training-representatives-v1',groups=groups,literal_zero_pcm_code=codes[-1],ordered_codes=codes,ordered_labels=labels.tolist()+[0],model=ex['model'],source_training_report_sha256=sha(training),source_export_report_sha256=sha(export),script_sha256=sha(Path(__file__)),model_calls=129,neural_cpu_gpu_comparison_performed=False,threshold_search=False,acceptance_accessed=False,promotion_qualified=False,limits='Actual INT8 scalar outputs versus known synthetic training class assignments on the same129calibration inputs. Already trained/calibration-consumed, not heldout recognition. Code235 corresponds to fixed0.91796875 on single windows; no app five-score decoder or streaming event metric here. No FP32 outputs, numerical agreement gate or retry. Cannot equate this crop classification with whole-wave training wake recall or attribute Melo misses to source alone.')
    with (R/'4090-matchbox-int8-training-representatives-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['ordered_codes','ordered_labels']}),flush=True)
if __name__=='__main__':main()
