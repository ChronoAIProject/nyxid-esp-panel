"""Keep whole sequences; change only command positive temporal reference."""
import json
from causal_sequence_batch_v1 import Sampler as OriginalSampler,digest,R
class Sampler(OriginalSampler):
    def __init__(self,seed=726):
        super().__init__(seed)
        p=json.loads((R/'4090-causal-command-prefix-reference-map-v1.json').read_text());assert p['mechanical_checks_passed']
        refs={r['id']:r for r in p['records']};assert len(refs)==432
        changed=0
        for row in self.rows:
            source=row['source']
            if source['id'] not in refs:continue
            ref=refs[source['id']]
            assert source['label']==1 and source['native_sha256']==ref['native_sha256'] and source['native_samples']==ref['native_samples']
            tail=row['container_end_samples']-row['lead_samples']-source['native_samples'];assert tail>=0
            end=row['lead_samples']+ref['automatic_prefix_end_sample']+tail
            assert row['lead_samples']<end<row['container_end_samples']
            # Original immutable material ledger is never edited. batch() reads
            # this private indexed copy for mask computation; waveform unchanged.
            self.by_index[row['order']]=dict(row,container_end_samples=end)
            changed+=1
        assert changed==5184
