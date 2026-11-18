"""Global-L2 nonadaptive SAM; exact restoration before original Adam update."""
import torch

def sam_gradients(model,optimizer,closure,rho):
    optimizer.zero_grad(set_to_none=True)
    loss=closure();assert torch.isfinite(loss);loss.backward()
    parameters=list(model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    norm=torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in parameters]),2)
    assert torch.isfinite(norm) and norm>0
    originals=[p.detach().clone() for p in parameters]
    with torch.no_grad():
        for p in parameters:p.add_(p.grad*(rho/(norm+1e-12)))
    optimizer.zero_grad(set_to_none=True)
    try:
        perturbed=closure();assert torch.isfinite(perturbed);perturbed.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    finally:
        with torch.no_grad():
            for p,v in zip(parameters,originals,strict=True):p.copy_(v)
    assert all(torch.equal(p,v) for p,v in zip(parameters,originals,strict=True))
    return loss.detach(),float(perturbed.detach()),float(norm.detach())
