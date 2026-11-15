"""Same original sampled descriptors, snap end within original offset envelope."""
from matchbox_long_batch_v1 import Sampler as OriginalSampler
class Sampler(OriginalSampler):
 def descriptors(self):
  descriptors=super().descriptors()
  for d in descriptors:
   if d['kind']!='full-bare':continue
   row=self.rows[d['index']];end=row['source_lead']+row['complete_source_and_rir_samples'];old_after=d['after']
   candidates=range(((end+256+479)//480)*480,end+1024+1,480)
   chosen=min(candidates,key=lambda x:(abs(x-(end+old_after)),x))
   d['original_after']=old_after;d['after']=chosen-end;d['start']=chosen-44800
   assert 0<=d['start']<=row['source_lead'] and 256<=d['after']<=1024 and chosen%480==0
  return descriptors
