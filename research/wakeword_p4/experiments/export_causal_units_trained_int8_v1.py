"""Trained cache INT8 export using real training-wave continuous states."""
import hashlib,json,os,sys
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1');os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
from causal_unit_sampler_v1 import Sampler,INDEX
from continuous_past_mfcc_v1 import ContinuousPastMFCC,FrameClock
from ordered_unit_event_policy_v1 import UnitEventPolicy
from transfer_causal_units_keras_v1 import build
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword');ROOT=B/'causal-generator-delay40-supervised-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def detail(d):return dict(shape=list(map(int,d['shape'])),dtype=str(d['dtype']),scale=float(d['quantization'][0]),zero=int(d['quantization'][1]))
def quantize(v,d):
 scale,zero=d['quantization'];assert scale>0
 return np.clip(np.rint(v/scale)+zero,-128,127).astype(np.int8)
def main():
 torch.set_num_threads(1);assert torch.cuda.is_available();tf.config.set_visible_devices([],'GPU');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 dest=ROOT/'int8.tflite';assert not dest.exists(),'Original export already exists'
 cp=ROOT/'final.pt';checkpoint=torch.load(cp,map_location='cpu',weights_only=False);assert checkpoint['step']==5000;sampler=Sampler();mapping=sampler.data['dense_to_original_label'];pt=CausalGeneratorUnitModel(len(mapping)).cuda().eval();pt.load_state_dict(checkpoint['model']);keras=build(checkpoint['model'],len(mapping));outputs=dict(keras.output);outputs['probabilities']=tf.keras.layers.Softmax(axis=-1,name='unit_probabilities')(outputs['logits']);keras=tf.keras.Model(keras.inputs,outputs)
 fp=B/'matchbox-grid-aligned-v1/frontend.pt';front=ContinuousPastMFCC(torch.load(fp,weights_only=True,map_location='cpu')).cuda().eval();representatives=[];waveforms=[];provenance=[];max_error=0.
 with torch.no_grad():
  for d in sampler.descriptors():
   r=sampler.rows[d['index']];assert sha(r['path'])==r['sha256'];pcm,sr=sf.read(r['path'],dtype='int16');assert sr==16000 and len(pcm)==r['samples'];assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256'];clock=FrameClock();frames,ends=clock.push(pcm);x=front(torch.tensor(frames.astype(np.float32)/32768,device='cuda')).T[None];cache=pt.encoder.initial_state(device='cuda');features=x.cpu().numpy().transpose(0,2,1)[:,:,None,:];waveforms.append((d,r,features,ends));selected=[]
   for start in range(0,x.shape[2]-2,3):
    chunk=x[:,:,start:start+3]
    if start%99==0:
     feed=dict(chunk=chunk.cpu().numpy().transpose(0,2,1)[:,:,None,:],**{f'cache{i}':c.cpu().numpy().transpose(0,2,1)[:,:,None,:] for i,c in enumerate(cache)});representatives.append(feed);selected.append(start)
     result=keras(feed,training=False);expected,nextcache=pt(chunk,cache);actual=result['logits'].numpy();expected=expected.cpu().numpy().transpose(0,2,1)[:,:,None,:];max_error=max(max_error,float(np.abs(actual-expected).max()));assert np.allclose(actual,expected,atol=2e-4,rtol=2e-4)
     for i,c in enumerate(nextcache):assert np.allclose(result[f'cache{i}_out'].numpy(),c.cpu().numpy().transpose(0,2,1)[:,:,None,:],atol=2e-4,rtol=2e-4)
    _,cache=pt(chunk,cache)
   provenance.append(dict(index=d['index'],path=r['path'],sha256=r['sha256'],pcm_sha256=r['pcm_sha256'],frames=len(frames),selected_frame_starts=selected))
 converter=tf.lite.TFLiteConverter.from_keras_model(keras);converter.optimizations=[tf.lite.Optimize.DEFAULT];converter.representative_dataset=lambda:iter(representatives);converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8];converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8;raw=converter.convert();dest.write_bytes(raw)
 it=Interpreter(model_path=str(dest),num_threads=1);it.allocate_tensors();runner=it.get_signature_runner();inputs=runner.get_input_details();outputs=runner.get_output_details();assert all(t['dtype'] in [np.int8,np.int32,np.int64] for t in it.get_tensor_details());assert outputs['probabilities']['quantization']==(1/256,-128);ops=[d['op_name'] for d in it._get_ops_details()];assert not any(x.startswith('Flex') for x in ops);records=[]
 for d,r,features,ends in waveforms:
  state={f'cache{i}':np.full(inputs[f'cache{i}']['shape'],inputs[f'cache{i}']['quantization'][1],np.int8) for i in range(6)};policy=UnitEventPolicy();events=[];digest=hashlib.sha256();calls=0
  for start in range(0,features.shape[1]-2,3):
   result=runner(chunk=quantize(features[:,start:start+3],inputs['chunk']),**state);calls+=1;prob=result['probabilities'][0,:,0,:];digest.update(prob.tobytes())
   for j,codes in enumerate(prob.astype(np.int16)+128):
    label=int(codes.argmax());original=mapping[label];token=original-1 if original else None
    if policy.push(token,int(codes[label])):events.append(int(ends[start+j]))
   for i in range(6):
    scale,zero=outputs[f'cache{i}_out']['quantization'];values=(result[f'cache{i}_out'].astype(np.float64)-zero)*scale;state[f'cache{i}']=quantize(values,inputs[f'cache{i}'])
  records.append(dict(index=d['index'],class_id=r['class_id'],carrier=r['carrier'],source_sha256=r['sha256'],event_samples=events,calls=calls,frames_consumed=calls*3,unscored_complete_frames=features.shape[1]-calls*3,probability_code_sha256=digest.hexdigest()))
 out=dict(schema='causal-delay40-trained-int8-export-v1',files={str(p):sha(p) for p in [Path(__file__),cp,fp,INDEX,Path('research/wakeword_p4/experiments/transfer_causal_units_keras_v1.py')]},model=dict(path=str(dest),bytes=len(raw),sha256=sha(dest)),calibration=provenance,representative_count=len(representatives),real_wave_float_transfer_max_abs_error=max_error,inputs={k:detail(v) for k,v in inputs.items()},outputs={k:detail(v) for k,v in outputs.items()},ops=ops,all_neural_tensors_integer=True,records=records,target_hits=sum(bool(r['event_samples']) for r in records if r['class_id']==1),target_count=4,negative_hits=sum(bool(r['event_samples']) for r in records if r['class_id']!=1),negative_count=20,policy=dict(confidence_code=128,max_span_frames=200,refractory_frames=200),qualification=False,limitations='First fixed training batch24 reused for calibration and diagnostic: not independent evaluation. External MFCC float, CUDA feature extraction; TFLite CPU1thread replay. Chunk3 fixed ABI leaves0–2complete frames unscored at end, no future/finish padding. Real cache reset uses zero points, feedback explicitly requantized. No P4 arena or full FAR claim. Original final model trained with40ms output delay; event timestamps are actual output time.')
 with (R/'4090-causal-delay40-trained-int8-export-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({k:out[k] for k in ['model','representative_count','real_wave_float_transfer_max_abs_error','target_hits','negative_hits']}))
if __name__=='__main__':main()
