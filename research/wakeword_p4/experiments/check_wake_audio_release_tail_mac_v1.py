"""Exercise the actual wake-task release tail with a pending call, without audio.

The extracted tail is compiled unchanged; pthread shims control worker-join
ordering. This is a teardown regression probe, not complete firmware execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE = 'components/panel_services/panel_voice.c'
PREAMBLE = r'''
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#define portMAX_DELAY 0
#define pdMS_TO_TICKS(x) (x)
static pthread_mutex_t capture_mutex=PTHREAD_MUTEX_INITIALIZER;
#define wake_lock (&capture_mutex)
static pthread_mutex_t event_mutex=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t event_cond=PTHREAD_COND_INITIALIZER;
static atomic_bool wake_listening;
static atomic_int_fast64_t wake_yield_until;
static void *wake_call;
static bool waiting,allow_wait,release_called,early_release;
int xSemaphoreTake(pthread_mutex_t *m,int timeout) {(void)timeout;return pthread_mutex_lock(m);}
int xSemaphoreGive(pthread_mutex_t *m) {return pthread_mutex_unlock(m);}
int64_t esp_timer_get_time(void) {return 123456;}
void vTaskDelay(int ticks) {
    if(ticks!=10)return;
    pthread_mutex_lock(&event_mutex);
    waiting=true;pthread_cond_broadcast(&event_cond);
    while(!allow_wait)pthread_cond_wait(&event_cond,&event_mutex);
    pthread_mutex_unlock(&event_mutex);
}
void product_audio_release(void) {
    pthread_mutex_lock(wake_lock);
    bool attached=wake_call!=NULL;
    pthread_mutex_unlock(wake_lock);
    pthread_mutex_lock(&event_mutex);
    early_release=attached;release_called=true;
    pthread_cond_broadcast(&event_cond);
    pthread_mutex_unlock(&event_mutex);
}
static void *run_tail(void *arg) {
    (void)arg;
'''
SUFFIX = r'''
    return NULL;
}
int main(int argc,char **argv) {
    alarm(5);
    bool pending=argc==2 && strcmp(argv[1],"pending")==0;
    int call=1;
    wake_call=pending?&call:NULL;
    atomic_store(&wake_listening,true);
    pthread_t thread;
    if(pthread_create(&thread,NULL,run_tail,NULL))return 10;
    pthread_mutex_lock(&event_mutex);
    while(!waiting && !release_called)pthread_cond_wait(&event_cond,&event_mutex);
    bool held=pending && waiting && !release_called;
    if(held) {
        // Represents call_task detaching only after peer/mic/speaker/tone join.
        // Taking the real capture mutex here also detects waiting under it.
        pthread_mutex_lock(wake_lock);wake_call=NULL;pthread_mutex_unlock(wake_lock);
    }
    allow_wait=true;pthread_cond_broadcast(&event_cond);
    pthread_mutex_unlock(&event_mutex);
    pthread_join(thread,NULL);
    bool ok=release_called && !early_release && (!pending || held)
        && !atomic_load(&wake_listening) && atomic_load(&wake_yield_until)==2123456;
    printf("{\"pending_call\":%s,\"held_until_detach\":%s,\"premature_release\":%s,\"contract_passed\":%s}\n",
        pending?"true":"false",held?"true":"false",early_release?"true":"false",ok?"true":"false");
    return ok?0:1;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-ref', default='WORKTREE')
    parser.add_argument('--expect', choices=('premature-release', 'safe-release'), required=True)
    parser.add_argument('--sanitizers', choices=('none', 'undefined', 'address-undefined'), default='undefined')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = (Path(SOURCE).read_bytes() if args.source_ref == 'WORKTREE' else
            subprocess.check_output(['git', 'show', args.source_ref + ':' + SOURCE]))
    text = data.decode()
    start = text.index('static void wake_task(')
    end = text.index('\nstatic void speaker_task(', start)
    body = text[start:end]
    final = '        vTaskDelay(pdMS_TO_TICKS(100));'
    end = body.rindex(final) + len(final)
    first = body.rindex('        atomic_store(&wake_listening,false);', 0, end)
    tail = body[first:end]
    assert tail.count('product_audio_release();') == 1
    args.output_dir.mkdir(exist_ok=False)
    unit = args.output_dir / 'release_tail.c'
    unit.write_text(PREAMBLE + tail + SUFFIX)
    exe = args.output_dir / 'release_tail'
    command = ['clang', '-std=c11', '-Wall', '-Wextra', '-Werror', '-pthread']
    if args.sanitizers != 'none':
        command.append('-fsanitize=' + args.sanitizers.replace('-', ','))
    command.extend([str(unit), '-o', str(exe)])
    built = subprocess.run(command, capture_output=True, text=True)
    (args.output_dir / 'compiler.log').write_text(built.stdout + built.stderr)
    assert built.returncode == 0, built.stderr
    results = []
    for case in ('empty', 'pending'):
        run = subprocess.run([str(exe), case], capture_output=True, text=True, timeout=10)
        (args.output_dir / (case + '.log')).write_text(run.stdout + run.stderr)
        row = json.loads(run.stdout)
        assert not run.stderr, run.stderr
        if case == 'pending' and args.expect == 'premature-release':
            assert run.returncode == 1 and row['premature_release'] and not row['contract_passed']
        else:
            assert run.returncode == 0 and row['contract_passed']
        results.append(dict(case=case, exit_code=run.returncode, **row))
    report = dict(schema='wake-audio-release-tail-probe-v1', source_ref=args.source_ref,
        source_sha256=hashlib.sha256(data).hexdigest(), tail_sha256=hashlib.sha256(tail.encode()).hexdigest(),
        probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        expected=args.expect, sanitizers=args.sanitizers, cases=results, expectation_met=True,
        microphone_or_speaker_used=False, hardware_operated=False,
        limits='Only the unchanged extracted wake-task release tail runs with deterministic pthread lifecycle shims. '
               'No I2S/codec/model/network/full FreeRTOS or long-call hardware acceptance.')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
