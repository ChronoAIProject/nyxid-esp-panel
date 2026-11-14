"""Inject allocation/rearm failures into actual wake-task exit branches on host."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE = 'components/panel_services/panel_voice.c'
PREAMBLE = r'''
#include <assert.h>
#include <setjmp.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void *allocations[32];
static int released, stopped;
static jmp_buf exited;
static atomic_bool wake_listening;
static char last_status[32];
static void *model,*models,*root,*aec,*agc,*capture,*mic,*ref,*clean,*adjusted,*detect;
static int wn_chunk=480;
static void *allocate(void) {
    for(int i=0;i<32;i++)if(!allocations[i]) {
        allocations[i]=malloc(16);assert(allocations[i]);return allocations[i];
    }
    abort();
}
static bool live(void *p) {
    if(!p)return false;
    for(int i=0;i<32;i++)if(allocations[i]==p)return true;
    return false;
}
static void tracked_free(void *p) {
    if(!p)return;
    for(int i=0;i<32;i++)if(allocations[i]==p) {
        free(p);allocations[i]=NULL;return;
    }
    assert(!"double free or foreign pointer");
}
#define free tracked_free
static void destroy_model(void *p) {assert(live(models)&&live(root));free(p);}
static int sample_rate(void *p) {(void)p;return 16000;}
static int channels(void *p) {(void)p;return 1;}
static const struct iface {
    void (*destroy)(void *);
    int (*get_samp_rate)(void *);
    int (*get_channel_num)(void *);
} iface={destroy_model,sample_rate,channels}, *wn=&iface;
static void aec_destroy(void *p) {free(p);}
static void esp_agc_close(void *p) {free(p);}
static void srmodel_host_deinit(void *p) {assert(!live(model)&&live(root));free(p);}
static bool prime_wake_model(const struct iface *w,void *m,void *d,int c) {
    (void)w;(void)m;(void)d;(void)c;return false;
}
static void wake_status(const char *s) {snprintf(last_status,sizeof(last_status),"%s",s);}
static void product_audio_release(void) {released++;}
static _Noreturn void vTaskDelete(void *p) {(void)p;stopped++;longjmp(exited,1);}
#define ESP_LOGE(...) ((void)0)
'''
SUFFIX = r'''
int main(int argc,char **argv) {
    assert(argc==2);int scenario=atoi(argv[1]);
    root=allocate();models=allocate();model=allocate();aec=allocate();agc=allocate();
    capture=allocate();mic=allocate();ref=allocate();clean=allocate();adjusted=allocate();detect=allocate();
    if(scenario==0){free(detect);detect=NULL;}
    if(scenario==1){destroy_model(model);model=NULL;}
    if(scenario==3){aec_destroy(aec);aec=NULL;}
    atomic_store(&wake_listening,false);
    if(!setjmp(exited))exercise(scenario);
    int retained=0;for(int i=0;i<32;i++)retained+=allocations[i]!=NULL;
    bool ok=!retained&&released==1&&stopped==1&&!strcmp(last_status,"unavailable");
    printf("{\"scenario\":%d,\"retained_allocations\":%d,\"audio_releases\":%d,\"task_exits\":%d,\"passed\":%s}\n",
           scenario,retained,released,stopped,ok?"true":"false");
    /* Release instrumentation leftovers after measuring the actual exit. */
    for(int i=0;i<32;i++)if(allocations[i])tracked_free(allocations[i]);
    return ok?0:1;
}
'''


def block(text, prefix):
    start = text.index(prefix)
    opening = text.index('{', start)
    depth = 1
    at = opening + 1
    while depth:
        depth += (text[at] == '{') - (text[at] == '}')
        at += 1
    return text[start:at]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-ref', default='WORKTREE')
    parser.add_argument('--expect', choices=('retained', 'released'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = (Path(SOURCE).read_bytes() if args.source_ref == 'WORKTREE' else
            subprocess.check_output(['git', 'show', args.source_ref + ':' + SOURCE]))
    text = data.decode()
    start = text.index('static void wake_task(void *arg)\n{')
    body = text[start:text.index('\nstatic void speaker_task(', start)]
    fragments = [block(body, 'if(!model || !aec || !agc'),
                 block(body, 'if(!model || !prime_wake_model'),
                 block(body, 'if(!aec){')]
    cleanup = body[body.index('\nunavailable:\n'):body.rindex('\n}')] if '\nunavailable:\n' in body else ''
    unit = PREAMBLE + '\nstatic void exercise(int scenario) {\n'
    for cases, fragment in [('scenario==0', fragments[0]),
                            ('scenario==1 || scenario==2', fragments[1]),
                            ('scenario==3', fragments[2])]:
        unit += f'if({cases}){{\n{fragment}\n}}\n'
    unit += '\nabort();\n' + cleanup + '\n}\n' + SUFFIX
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cpath = args.output_dir / 'failure_cleanup.c'
    cpath.write_text(unit)
    exe = args.output_dir / 'failure_cleanup'
    built = subprocess.run(['clang', '-std=c11', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=undefined', str(cpath), '-o', str(exe)],
                           capture_output=True, text=True)
    (args.output_dir / 'compiler.log').write_text(built.stdout + built.stderr)
    assert built.returncode == 0, built.stderr
    rows = []
    for case in range(4):
        run = subprocess.run([str(exe), str(case)], capture_output=True, text=True, timeout=10)
        (args.output_dir / f'case-{case}.log').write_text(run.stdout + run.stderr)
        assert not run.stderr, run.stderr
        row = dict(exit_code=run.returncode, **json.loads(run.stdout))
        assert run.returncode in (0, 1) and row['audio_releases'] == row['task_exits'] == 1
        rows.append(row)
    if args.expect == 'released':
        assert all(r['passed'] for r in rows), rows
    else:
        assert rows[0]['passed'] and all(r['retained_allocations'] > 0 for r in rows[1:]), rows
    report = dict(schema='wake-failure-cleanup-host-v1', source_ref=args.source_ref,
                  source_sha256=hashlib.sha256(data).hexdigest(),
                  extracted_branch_sha256=[hashlib.sha256(f.encode()).hexdigest() for f in fragments],
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  expect=args.expect, cases=rows, expectation_met=True,
                  no_hardware_operation=True, no_audio_io=True,
                  limits='Actual extracted failure branches with tracked host allocations and resource API shims. Not a full task, ESP allocator/AEC implementation, hardware leak reproduction or cause of observed restarts.')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
