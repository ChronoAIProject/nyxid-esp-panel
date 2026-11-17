"""CUDA functional gate only: no training, evaluation audio or qualification."""
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path('research/wakeword_p4/training').resolve()))
from causal_binary40_v1 import Binary40, geometry


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(726)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
    model = Binary40().cuda().eval()
    x = torch.rand(2, 40, 846, device='cuda')*26
    checks = []
    with torch.no_grad():
        full, final = model(x, model.initial_state(2, 'cuda'))
        for size in (3, 6, 37, 128, 846):
            states = model.initial_state(2, 'cuda')
            pending = x[:, :, :0]
            outputs = []
            for start in range(0, x.shape[2], size):
                pending = torch.cat((pending, x[:, :, start:start+size]), dim=2)
                count = pending.shape[2]//3*3
                if count:
                    y, states = model(pending[:, :, :count], states)
                    outputs.append(y)
                    pending = pending[:, :, count:]
            actual = torch.cat(outputs, dim=2)
            error = float((actual-full).abs().max())
            assert torch.allclose(actual, full, atol=1e-4, rtol=1e-5), (size, error)
            assert all(torch.allclose(a, b, atol=1e-4, rtol=1e-5) for a,b in zip(states,final))
            assert pending.shape[2] == 0
            checks.append(dict(host_chunk_frames=size, max_abs_error=error, outputs=actual.shape[2]))
        changed = x.clone()
        changed[:, :, 423:] = torch.rand_like(changed[:, :, 423:])*26
        causal, _ = model(changed, model.initial_state(2, 'cuda'))
        assert torch.equal(causal[:, :, :141], full[:, :, :141])
    model.train()
    logits, _ = model(x, model.initial_state(2, 'cuda'))
    labels = torch.zeros_like(logits)
    labels[:, :, -9:] = 1
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert geometry()['receptive_field_frames'] == 306
    paths = [Path(__file__), Path('research/wakeword_p4/training/causal_binary40_v1.py')]
    report = dict(schema='causal-binary40-functional-v1', gpu=torch.cuda.get_device_name(),
                  torch_version=torch.__version__, geometry=geometry(),
                  parameters=sum(p.numel() for p in model.parameters()),
                  files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  chunk_checks=checks, future_perturbation_prior141_outputs_exact=True,
                  gradient_parameters_finite=True, analytic_loss=float(loss.detach()),
                  limitations='Random untrained CUDA function only; no optimizer updates, audio, accuracy, INT8/export, P4 or human claims. Full/chunk parity does not prove training startup equals physical primed audio. Same306-frame span does not mean same learned function as C161.')
    out=Path('research/wakeword_p4/results/4090-causal-binary40-functional-v1.json')
    with out.open('x') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
