"""Queue one frozen float diagnostic after authenticated broad training audit."""
import json
import subprocess
import time
from pathlib import Path

from continue_broadcast_residual_pipeline_v1 import ENV, R, publish, python, stage


def main():
    launch = json.loads(Path('.local/wakeword/broadcast-residual-float-context-launch-v1.json').read_text())
    proc = Path('/proc') / str(launch['predecessor_pid']) / 'stat'
    while proc.exists():
        fields = proc.read_text().rsplit(')', 1)[1].split()
        assert fields[19] == launch['predecessor_starttime'], 'PID reused; no restart'
        if fields[0] == 'Z':
            break
        time.sleep(15)
    for name in ['consumed-independent-review', 'broad-fit', 'broad-fit-review', 'bare-transitions-review']:
        receipt = json.loads((R / f'4090-broadcast-residual-{name}-execution-v1.json').read_text())
        assert receipt['actual_exit_code'] == 0, name
    source = R / '4090-broadcast-residual-joint-context-original-v1.json'
    assert subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:' + str(source)]) == source.read_bytes()
    subprocess.run(python('broadcast_residual_float_context_v1.py', 'plan'), check=True, env=ENV)
    publish([R / '4090-broadcast-residual-float-context-plan-v1.json'],
            'research: lock frozen broadcast residual float diagnostic on three complete consumed contexts')
    receipts = [stage('float-context', python('broadcast_residual_float_context_v1.py', 'run')),
                stage('float-context-review', python('review_broadcast_residual_float_context_v1.py'))]
    outputs = [R / f'4090-broadcast-residual-float-context-{suffix}-v1.{ext}' for suffix, ext in
               [('original', 'json'), ('review', 'json'), ('evidence', 'tar.gz')]]
    publish([*receipts, *outputs],
            'research: preserve frozen broadcast residual float and INT8 context comparison with independent review')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    body = Path('/tmp/nyx-broadcast-residual-float-context-completed-v1.md')
    body.write_text('广播残差最终冻结权重的三组完整连说上下文 CUDA4090 浮点对照及独立事件/归档审核实际 exit0，'
                    '原始 float/INT8 概率与逐条结果已推送 ' + commit + '。零优化更新，未改阈值、未重新导出或校准。'
                    '仅用于区分这三条已消费流的量化前后现象，不解释其他声音/背景原因，不代表 P4 合格；C161 保持。\n')
    subprocess.run(['gh', 'issue', 'comment', '18', '--repo', 'ChronoAIProject/nyxid-esp-panel',
                    '--body-file', str(body)], check=True, env=ENV)


if __name__ == '__main__':
    main()
