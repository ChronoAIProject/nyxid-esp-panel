"""Prelocked random-input causal/cache/gradient checks; no recognition or updates."""
import hashlib, json, subprocess, sys
from pathlib import Path

R=Path('research/wakeword_p4/results')
LOCK=R/'4090-broadcast-residual-mechanical-plan-v1.json'
MODEL=Path('research/wakeword_p4/training/causal_broadcast_residual_v1.py')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def plan():
    paths=[Path(__file__), MODEL, Path(__file__).with_name('broadcast_residual_reference_review_v1.md')]
    spec=dict(files={str(p):sha(p) for p in paths}, seed=726, batch=2, frames=300,
              chunk_frames=3, prefix_frames=150, optimizer_updates=0,
              policy='Untrained adaptation only. Random input full/chunk outputs, all states, gradients, '
              'future perturbation and future-gradient exclusion in train/eval modes. No speech scoring, '
              'calibration, synthesis, optimizer or heldout access. Not reference BC-ResNet reproduction '
              'or evidence of Hye Nyx quality/P4 feasibility. No BN/SSN/Swish/global time pooling.')
    with LOCK.open('x') as f: json.dump(spec,f,indent=2);f.write('\n')

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for name,h in p['files'].items(): assert sha(name)==h,name
    import torch
    sys.path.insert(0,str(MODEL.parent.resolve()))
    from causal_broadcast_residual_v1 import CausalBroadcastResidual
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(p['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
    model=CausalBroadcastResidual().cuda()
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    forbidden=(torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm,
               torch.nn.GroupNorm, torch.nn.Dropout, torch.nn.Dropout2d)
    assert not any(isinstance(m,forbidden) for m in model.modules())
    x=torch.randn(p['batch'],40,p['frames'],device='cuda')
    rows=[]
    for training in [True,False]:
        model.train(training)
        model.zero_grad(set_to_none=True)
        full,full_states=model(x,model.initial_state(p['batch'],'cuda'))
        full.square().mean().backward()
        grads={k:v.grad.detach().clone() for k,v in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        states=model.initial_state(p['batch'],'cuda');parts=[]
        for i in range(0,p['frames'],p['chunk_frames']):
            y,states=model(x[:,:,i:i+p['chunk_frames']],states);parts.append(y)
        streamed=torch.cat(parts,dim=2)
        torch.testing.assert_close(full,streamed,atol=2e-4,rtol=2e-4)
        for a,b in zip(full_states,states,strict=True):
            torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
        streamed.square().mean().backward()
        gradient_error=0.
        for k,v in model.named_parameters():
            assert torch.isfinite(v.grad).all() and torch.isfinite(grads[k]).all()
            torch.testing.assert_close(v.grad,grads[k],atol=2e-4,rtol=2e-4)
            gradient_error=max(gradient_error,float((v.grad-grads[k]).abs().max()))
        changed=x.clone();changed[:,:,p['prefix_frames']:]+=10*torch.randn_like(changed[:,:,p['prefix_frames']:])
        with torch.no_grad():
            altered,_=model(changed,model.initial_state(p['batch'],'cuda'))
            prefix=full[:,:,:p['prefix_frames']//3]
            assert torch.equal(prefix,altered[:,:,:p['prefix_frames']//3])
            again,reset_states=model(x,model.initial_state(p['batch'],'cuda'))
            assert torch.equal(again,full)
            assert all(torch.equal(a,b) for a,b in zip(reset_states,full_states,strict=True))
        differentiable=x.clone().requires_grad_()
        causal,_=model(differentiable,model.initial_state(p['batch'],'cuda'))
        derivative=torch.autograd.grad(causal[:,:,:p['prefix_frames']//3].sum(),differentiable)[0]
        assert torch.count_nonzero(derivative[:,:,p['prefix_frames']:])==0
        rows.append(dict(training=training,output_max_error=float((full-streamed).detach().abs().max()),
                         state_max_error=max(float((a-b).detach().abs().max()) for a,b in zip(full_states,states)),
                         gradient_max_error=gradient_error,prefix_future_perturbation_exact=True,
                         future_gradient_nonzero=0,fresh_reset_exact=True))
    assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items())
    assert sum(t.numel() for t in model.initial_state())==model.geometry()['state_elements']
    out=dict(all_passed=True,plan_sha256=sha(LOCK),geometry=model.geometry(),checks=rows,
             gpu=torch.cuda.get_device_name(),torch=torch.__version__,optimizer_updates=0,
             weights_unchanged=True,policy=p['policy'])
    with (R/'4090-broadcast-residual-mechanical-original-v1.json').open('x') as f:
        json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out),flush=True)

if __name__=='__main__': {'plan':plan,'run':run}[sys.argv[1]]()
