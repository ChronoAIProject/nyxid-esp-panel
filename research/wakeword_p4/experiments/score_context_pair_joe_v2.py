"""Repair metadata-only scoring of already completed Joe traces; no reinference."""
import ast
import json
from pathlib import Path
import subprocess
import replay_context_pair_coverage_v1 as parent
R=parent.R

def main():
    own=Path(__file__);assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(own)])==own.read_bytes()
    fn=next(n for n in ast.parse(Path(parent.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='score')
    class Adapter(ast.NodeTransformer):
        def visit_Subscript(self,n):
            self.generic_visit(n)
            if isinstance(n.value,ast.Name) and n.value.id=='row' and isinstance(n.slice,ast.Constant) and n.slice.value=='carrier':
                return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='row',ctx=ast.Load()),attr='get',ctx=ast.Load()),args=[ast.Constant('carrier'),ast.Constant('whole-utterance-unclassified')],keywords=[]),n)
            return n
    env=dict(vars(parent));exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(own),'exec'),env);env['score']('joe')
    log=Path('.local/wakeword/context-pair-coverage-execution-v1/joe-score.log');result=R/'4090-context-pair-joe-coverage-comparison-v1.json'
    receipt=dict(scoring_exit_code=0,inference_repeated=False,original_score_exit_code=1,original_failure_log=log.read_text(),original_failure_log_sha256=parent.common.sha(log),scorer_sha256=parent.common.sha(own),source_scorer_sha256=parent.common.sha(Path(parent.__file__)),comparison_sha256=parent.common.sha(result),scope='Only missing carrier metadata mapped to whole-utterance-unclassified. No source labels/bytes/events/threshold modified. Joe generated boundaries do not identify a wake-prefix endpoint.')
    (R/'4090-context-pair-joe-score-repair-v2.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
