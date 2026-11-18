"""Wait for the registered pipeline, then audit frozen training support once."""
import json, subprocess, time
from pathlib import Path
from continue_local_stem_pipeline_v1 import R, python, publish, stage, ENV

def main():
    launch_path=Path('.local/wakeword/local-stem-broad-fit-launch-v1.json')
    launch=json.loads(launch_path.read_text())
    proc=Path('/proc')/str(launch['pipeline_pid'])/'stat'
    while proc.exists():
        fields=proc.read_text().rsplit(')',1)[1].split()
        assert fields[19]==launch['pipeline_starttime'], 'Pipeline PID reused; no restart'
        if fields[0]=='Z': break
        time.sleep(15)
    receipt=R/'4090-local-stem-consumed-independent-review-execution-v1.json'
    corrected=R/'4090-local-stem-consumed-independent-review-corrected-execution-v1.json'
    if corrected.exists():
        assert json.loads(receipt.read_text())['actual_exit_code']==1
        receipt=corrected
    assert json.loads(receipt.read_text())['actual_exit_code']==0
    review=R/'4090-local-stem-consumed-review-v1.json'
    assert json.loads(review.read_text())['all_passed']
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(review)])==review.read_bytes()
    subprocess.run(python('audit_local_stem_broad_fit_v1.py','plan'),check=True,env=ENV)
    plan=R/'4090-local-stem-broad-fit-plan-v1.json'
    publish([plan],'research: lock complete local stem training support audit without optimization')
    receipts=[stage('broad-fit',python('audit_local_stem_broad_fit_v1.py','run')),
              stage('broad-fit-review',python('review_local_stem_broad_fit_v1.py')),
              stage('bare-transitions-review',python('review_local_stem_bare_transitions_v1.py'))]
    outputs=[R/name for name in ['4090-local-stem-broad-fit-original-v1.json',
              '4090-local-stem-broad-fit-traces-v1.npz','4090-local-stem-broad-fit-review-v1.json',
              '4090-local-stem-bare-transitions-review-v1.json']]
    publish([*receipts,*outputs],'research: preserve exhaustive local stem training support and independent mask transitions')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    body=Path('/tmp/nyx-local-stem-broad-fit-completed-v1.md')
    body.write_text('局部频率stem最终冻结权重的完整31644条原始bare训练来源GPU零更新审计及独立事件/时间mask/逐来源得失复核实际exit0，原始概率与报告已推送 '+commit+'。这是训练内覆盖诊断，不代表独立泛化或P4合格，未调整阈值或重训；C161保持，下一步结合完整cohort/context逐句比较解读。\n')
    subprocess.run(['gh','issue','comment','18','--repo','ChronoAIProject/nyxid-esp-panel','--body-file',str(body)],check=True,env=ENV)

if __name__=='__main__': main()
