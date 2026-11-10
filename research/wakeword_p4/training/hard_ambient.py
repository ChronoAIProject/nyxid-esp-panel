"""Sample previously audited real-speech hard negatives from training streams."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HardAmbient:
    def __init__(self, audit_path, features, frames):
        self.audit_path=Path(audit_path);features=Path(features)
        audit=json.loads(self.audit_path.read_text())
        provenance=json.loads((features/'base-provenance.json').read_text())
        if (audit['role']!='Audited training-only hard ambient windows' or
                audit['validation_audio_accessed'] or audit['independent_final_audio_accessed'] or
                audit['context_frames']!=frames or
                provenance['source_human_index_sha256']!=audit['feature_index_sha256']):
            raise ValueError('Hard negatives require matching audited training-only source')
        rows={str(r['speaker']):r for r in json.loads((features/'human.json').read_text())}
        self.candidates=audit['candidates'];self.data={}
        if not self.candidates:raise ValueError('Empty hard-negative audit')
        for c in self.candidates:
            row=rows[str(c['speaker'])]
            if (c['split']!='training' or row['split']!='training' or c['quarantined'] or
                    c['feature_sha256']!=row['feature_sha256'] or c['jitter_frames']!=audit['jitter_frames'] or
                    c['frame_end']-c['frame_start']!=frames):
                raise ValueError('Changed, quarantined or non-training hard-negative record')
            if c['speaker'] not in self.data:
                # Existing composite datasets intentionally link audited source directories.
                path=features/row['path']
                if '..' in Path(row['path']).parts or Path(row['path']).is_absolute():
                    raise ValueError('Escaped canonical training feature path')
                if sha(path)!=c['feature_sha256']:raise ValueError('Changed hard-negative stream')
                data=np.load(path,mmap_mode='r',allow_pickle=False)
                if data.dtype!=np.uint16 or data.shape!=(row['frames'],40):
                    raise ValueError('Invalid hard-negative feature shape')
                self.data[c['speaker']]=data
            if not frames<=c['frame_end']<=len(self.data[c['speaker']]):
                raise ValueError('Hard-negative window outside source')
        self.frames=frames
        categories=Counter('human:'+str(rows[str(c['speaker'])].get('category','unspecified'))
                           for c in self.candidates)
        self.report=dict(audit_sha256=sha(self.audit_path),candidate_count=len(self.candidates),
                         speaker_count=len(self.data),jitter_frames=audit['jitter_frames'],
                         mining_model_sha256=audit['mining_model_sha256'],
                         source_feature_index_sha256=audit['feature_index_sha256'],
                         category_probabilities={k:v/len(self.candidates) for k,v in categories.items()},
                         label_policy=audit['label_policy'])

    def sample(self, count, rng):
        output=[]
        for _ in range(count):
            candidate=self.candidates[int(rng.integers(len(self.candidates)))]
            data=self.data[candidate['speaker']];jitter=candidate['jitter_frames']
            lo=max(self.frames,candidate['frame_end']-jitter)
            hi=min(len(data),candidate['frame_end']+jitter)
            end=int(rng.integers(lo,hi+1))
            output.append(data[end-self.frames:end])
        return np.asarray(output)
