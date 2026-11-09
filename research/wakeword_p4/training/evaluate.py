"""Choose a threshold on validation only, then score untouched held-out audio."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from scipy.stats import chi2
import soundfile as sf

from features import features, render

THRESHOLDS = [.50, .70, .80, .90, .95, .97, .98, .985, .99, .992, .994]
SMOOTH = 5


class Detector:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.interpreter = Interpreter(model_path=str(self.model), experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.scale, self.zero = self.input['quantization']
        self.output_scale, self.output_zero = self.output['quantization']

    def predict(self, raw):
        values = np.clip(raw.astype(np.float32)/25.6, 0, 26)
        values = np.clip(np.rint(values/self.scale)+self.zero, -128, 127).astype(np.int8)
        result = []
        for at in range(0, len(values)-2, 3):
            self.interpreter.set_tensor(self.input['index'], values[None, at:at+3])
            self.interpreter.invoke()
            score = self.interpreter.get_tensor(self.output['index'])[0,0]
            result.append((int(score)-self.output_zero)*self.output_scale)
        return np.asarray(result, dtype=np.float32)


def smooth(scores):
    return np.convolve(scores, np.ones(SMOOTH)/SMOOTH, mode='valid') if len(scores)>=SMOOTH else np.array([])


def events(scores, threshold):
    # Five consecutive 30 ms scores, rising-edge latch, and >=2 s refractory.
    count = 0
    armed = True
    last = -1000
    for index, value in enumerate(scores):
        if value < threshold*.5 and index-last>=67:
            armed=True
        if armed and value>=threshold and index-last>=67:
            count+=1;last=index;armed=False
    return count


def tts_scores(detector, records, dataset, split):
    result = []
    group = [r for r in records if r['split']==split]
    started = time.monotonic()
    for index,item in enumerate(group):
        for variant in (0,1):
            seed = int(hashlib.sha256(item['path'].encode()).hexdigest()[:8],16)+variant
            rng = np.random.default_rng(seed)
            pcm,lead = render(item,dataset,rng,augmented=variant>0)
            detector.reset()
            raw = features(pcm)
            scores = smooth(detector.predict(raw))
            # Raw feature j ends at (j+3)*10 ms; stream step k ends at
            # (3*k+5)*10 ms. Smoothing consumes four more stream steps.
            times = (3*(np.arange(len(scores))+SMOOTH-1)+5)*.01
            end = (lead+item['keyword_end'])/16000 if item['label'] else None
            valid = (times>=end-.10)&(times<=end+.80) if item['label'] else np.ones(len(scores),dtype=bool)
            score = float(np.max(scores[valid])) if np.any(valid) else 0
            result.append({'source':item['path'],'label':item['label'],'text':item['text'],'variant':variant,
                           'max_score':score,'keyword_end':end,
                           'scores':scores.tolist(),'times':times.tolist()})
        if index%120==0:
            print(json.dumps({'phase':split,'clips':index,'total':len(group),'seconds':round(time.monotonic()-started,1)}),flush=True)
    return result


def human_scores(detector, records, feature_root, split, output):
    result=[]
    for item in records:
        if item['split']!=split:continue
        raw=np.load(feature_root/item['path'],mmap_mode='r')
        detector.reset()
        scores=smooth(detector.predict(raw))
        np.save(output/f'human-{split}-{item["speaker"]}.npy',scores)
        hours=len(raw)*.01/3600
        result.append({'speaker':item['speaker'],'hours':hours,'events':{str(t):events(scores,t) for t in THRESHOLDS}})
        print(json.dumps({'phase':split+'_human','speaker':item['speaker'],'hours':round(hours,3),'events_097':events(scores,.97)}),flush=True)
    return result


def summarize(clips, human, threshold):
    positive=[r for r in clips if r['label']]
    negative=[r for r in clips if not r['label']]
    hit=sum(r['max_score']>=threshold for r in positive)
    false=sum(r['max_score']>=threshold for r in negative)
    count=sum(r['events'][str(threshold)] for r in human)
    hours=sum(r['hours'] for r in human)
    latency=[]
    for r in positive:
        above=np.flatnonzero(np.asarray(r['scores'])>=threshold)
        valid=[i for i in above if r['keyword_end']-.10<=r['times'][i]<=r['keyword_end']+.80]
        if valid:latency.append((r['times'][valid[0]]-r['keyword_end'])*1000)
    return {'threshold':threshold,'positive_clips':len(positive),'detected':hit,'missed':len(positive)-hit,
            'recall':hit/len(positive),'hard_negative_clips':len(negative),'hard_negative_false':false,
            'hard_negative_false_rate':false/len(negative),'human_hours':hours,'human_false_events':count,
            'human_false_per_hour':count/hours,'human_false_per_hour_upper_95':float(.5*chi2.ppf(.95,2*(count+1))/hours),
            'latency_ms_median':float(np.median(latency)) if latency else None,
            'latency_ms_p95':float(np.quantile(latency,.95)) if latency else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--mac-eval',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'report.json').exists(),'Use a fresh evaluation directory'
    detector=Detector(args.model)
    records=[json.loads(line) for line in (args.dataset/'manifest.jsonl').read_text().splitlines()]
    human=json.loads((args.features/'human.json').read_text())
    val=tts_scores(detector,records,args.dataset,'validation')
    vh=human_scores(detector,human,args.features,'validation',args.output)
    grid=[summarize(val,vh,t) for t in THRESHOLDS]
    # Threshold selection is completed and recorded before test data is read.
    eligible=[r for r in grid if r['human_false_events']==0 and r['hard_negative_false_rate']<=.01]
    if eligible:
        selected=max(eligible,key=lambda r:(r['recall'],-r['threshold']))
        criterion='Zero observed validation human events and <=1% hard-negative clip false detections; maximize recall'
    else:
        selected=min(grid,key=lambda r:(r['human_false_events'],r['hard_negative_false_rate'],-r['recall']))
        criterion='No threshold met desired validation criteria; report best fallback without claiming acceptance'
    (args.output/'threshold.json').write_text(json.dumps({'selection':selected,'criterion':criterion,'grid':grid},indent=2)+'\n')
    print(json.dumps({'selected':selected,'criterion':criterion}),flush=True)
    test=tts_scores(detector,records,args.dataset,'testing')
    th=human_scores(detector,human,args.features,'testing',args.output)
    final=summarize(test,th,selected['threshold'])
    final['clean']=summarize([r for r in test if r['variant']==0],th,selected['threshold'])
    final['augmented']=summarize([r for r in test if r['variant']==1],th,selected['threshold'])
    mac=[]
    for line in (args.mac_eval/'manifest.jsonl').read_text().splitlines():
        item=json.loads(line);pcm,rate=sf.read(args.mac_eval/item['path'],dtype='int16');assert rate==16000
        for variant in (0,1):
            rng=np.random.default_rng(int(hashlib.sha256(item['path'].encode()).hexdigest()[:8],16)+variant)
            lead=np.zeros(28800,dtype=np.float32)
            wave=np.concatenate((lead,pcm.astype(np.float32),np.zeros(16000,dtype=np.float32)))
            if variant:
                noise=rng.normal(0,300,len(wave));wave=wave*.3+noise
            detector.reset();scores=smooth(detector.predict(features(wave)))
            mac.append({'source':item['path'],'voice':item['voice'],'text':item['text'],'label':item['label'],
                        'variant':variant,'max_score':float(np.max(scores))})
    mp=[r for r in mac if r['label']];mn=[r for r in mac if not r['label']]
    mac_summary={'positive_clips':len(mp),'detected':sum(r['max_score']>=selected['threshold'] for r in mp),
                 'negative_clips':len(mn),'false_detections':sum(r['max_score']>=selected['threshold'] for r in mn)}
    mac_summary['recall']=mac_summary['detected']/len(mp)
    report={'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'sliding_window':SMOOTH,
            'threshold':selected['threshold'],'selection_criterion':criterion,'validation':selected,
            'testing':final,'independent_mac_tts':mac_summary,
            'limits':['No real human positive wake-word recordings; synthetic positive recall does not establish real user recall.',
                      'Human negatives are English audiobooks; Chinese/TV/music/household noise not yet evaluated.',
                      'Human false-activation exposure includes 200 ms inserted silence at utterance boundaries.',
                      'First trained research candidate, not a production acceptance result.']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'clip-details.json').write_text(json.dumps({'validation':val,'testing':test,'mac':mac})+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
