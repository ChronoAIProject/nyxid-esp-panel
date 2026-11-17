"""Full per-batch matched sampling audit, usable before optimizer updates."""
import numpy as np
from causal_source_balanced_batch_v1 import digest


def audit(original, original_descriptors, treatment, descriptors):
    assert len(original_descriptors) == len(descriptors) == 128
    assert [v.shape for v in original] == [v.shape for v in treatment] == [(128,846,40),(128,282),(128,282)]
    ps = [i for i,r in enumerate(original_descriptors) if r['kind']=='existing_broad_bare_positive'][:4]
    ns = [i for i,r in enumerate(original_descriptors) if r['kind']=='source_balanced_bare_negative'][:4]
    changed = sorted(ps+ns)
    assert len(ps)==len(ns)==4 and len(set(changed))==8
    assert [i for i in range(128) if original_descriptors[i]!=descriptors[i]] == changed
    unchanged = [i for i in range(128) if i not in changed]
    for i in unchanged:
        assert original_descriptors[i]==descriptors[i]
        assert all(np.array_equal(a[i],b[i]) for a,b in zip(original,treatment,strict=True))
    for i,j in zip(ps,ns,strict=True):
        a,b=descriptors[i],descriptors[j]
        assert a['kind']=='paired_carrier_positive' and a['label']==1
        assert b['kind']=='paired_carrier_nick_negative' and b['label']==0
        for key in ['conditioning_id','speed','carrier','variant','positive_offset','negative_offset']:
            assert a[key]==b[key]
    assert sum(r['label'] for r in descriptors)==32
    assert all(np.isfinite(v).all() for v in treatment)
    for i,r in enumerate(descriptors):
        assert np.isclose(treatment[1][i].sum(),r['label'])
        assert np.isclose(treatment[2][i].sum(),1)
        assert not np.any((treatment[1][i]>0)&(treatment[2][i]>0))
    return dict(changed_slots=changed,unchanged_rows=120,
                original_sha256=[digest(v) for v in original],
                treatment_sha256=[digest(v) for v in treatment],
                unchanged_sha256=[digest(v[unchanged]) for v in treatment])
