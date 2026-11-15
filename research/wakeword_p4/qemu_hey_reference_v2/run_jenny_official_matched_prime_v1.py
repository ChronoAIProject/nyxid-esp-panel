"""Same original official firmware, continuous total81600 zero PCM warmup."""
import ast,json,sys
from pathlib import Path
import run_nearword_jenny_default_v1 as parent
ROOT=parent.ROOT;SOURCE=parent.SOURCE;sha=parent.sha
PLAN=ROOT/'results/4090-jenny-official-matched-prime-plan-v1.json'
EXTRA=81600-63*512
assert EXTRA==49344

def register():
    original=parent.PLAN;p=json.loads(original.read_text())
    for f,h in p['files'].items():assert sha(f)==h,f
    p['files'].update({str(f):sha(f) for f in [Path(__file__),Path(parent.__file__),original]})
    p.update(schema='jenny-official-matched-prime-plan-v1',output='.local/wakeword/jenny-official-matched-prime-v1',firmware_prime_samples=32256,additional_continuous_zero_samples=EXTRA,total_prime_samples=81600,policy='Same original396 Jenny whole synthetic inputs and12 complete9.614h background, fixed original native DEFAULT official HiESP. Existing unmodified firmware63x512zero prime then host input49344zero PCM plus entire unchanged original PCM, one continuous detector, no transition reset, finalpadding only at whole stream end. Totalzero warmup81600 matches candidates, retain pending native block and all prime returns/events; app ignores only events whose block-return clock is before/equal real-source start. Native512chunk vs480candidate is intrinsic and reported, not forced numerical equality. Original source bytes/hashes preserved; input/CRC additionally cover prefix. Same consumed development only, no newTTS/training/thresholdsearch/finalaccess/promotion. Previous32256prime reference retained, no claims of universal P4 parity or phonetic truth.')
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(sha(PLAN))

def case():
    tree=ast.parse(SOURCE.read_text());counts=dict(plan=0,insert=0,report=0)
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='research/wakeword_p4/results/4090-hey-official-development-execution-v1.json':counts['plan']+=1;return ast.copy_location(ast.Constant(str(PLAN)),n)
            return n
        def visit_Expr(self,n):
            self.generic_visit(n)
            if ast.unparse(n).startswith('a.output.mkdir('):
                counts['insert']+=1
                return ast.parse('source_samples=samples\nsource_raw_sha256=hashlib.sha256(raw).hexdigest()\nextra_prime_samples=49344\nraw=bytes(extra_prime_samples*2)+raw\nsamples+=extra_prime_samples').body+[n]
            return n
        def visit_Assign(self,n):
            self.generic_visit(n)
            if any(isinstance(t,ast.Name) and t.id=='report' for t in n.targets):
                counts['report']+=1
                return [n]+ast.parse("report.update(schema='jenny-official-matched-prime-original-v1',source_input_samples=source_samples,source_pcm_sha256=source_raw_sha256,input_pcm_sha256=hashlib.sha256(raw).hexdigest(),firmware_prime_samples=32256,additional_prime_samples=extra_prime_samples,total_prime_samples=81600,continuous_model=True,transition_reset=False,real_events=[dict(e,relative_sample=e['sample_end']-extra_prime_samples) for e in events if e['sample_end']>extra_prime_samples],additional_priming_events=[e for e in events if e['sample_end']<=extra_prime_samples],limits='Original source intact with continuous49344zero PCM prefix after firmware32256warmup; total81600, pending block retained, no transition reset. Native512clock, all events retained. Consumed development only, no P4/phonetic truth claim.')").body
            return n
    tree=Adapter().visit(tree);assert counts==dict(plan=1,insert=1,report=1),counts
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE),'exec'),dict(__name__='__main__',__file__=__file__))

def run():
    # Reuse sequential full-input transport/copy verification; fresh paths only.
    original=Path(parent.__file__)
    assert sha(original)==json.loads(PLAN.read_text())['files'][str(original)]
    parent.PLAN=PLAN;parent.__file__=__file__
    # Parent temporary names are now vacant after prior successful copy/cleanup.
    parent.run()

if __name__=='__main__':
    action=sys.argv.pop(1);{'register':register,'case':case,'run':run}[action]()
