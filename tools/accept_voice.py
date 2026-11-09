#!/usr/bin/env python3
"""Physical Mac-speaker/board-microphone voice acceptance, using a saved owner CLI session.

Scenarios full, remote, and boundaries open and hang up their own calls. Full
temporarily mutes the call; boundaries temporarily changes the saved wake switch.
No Key, grants, Bot selection, volume or image is changed. See docs/DEBUGGING.md.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import uuid


class Acceptance:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.report = {'scenario': args.scenario, 'events': []}
        self.session = None
        self.remote_id = None
        self.initial = None
        self.before = 0
        self.player = None
        self.cached_at, self.cached = 0, None

    def event(self, name, **fields):
        value = {'name': name, 'at': round(time.monotonic() - self.started, 3), **fields}
        self.report['events'].append(value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def cli(self, *arguments):
        command = ['nyxid', *arguments, '--output', 'json']
        if self.args.profile:
            command += ['--profile', self.args.profile]
        return json.loads(subprocess.check_output(command, text=True, timeout=45))

    def proxy(self, path, body=None):
        command = ['proxy', 'request', self.args.service, path]
        if body is not None:
            command += ['-m', 'POST', '-H', 'Content-Type:application/json', '-d', json.dumps(body)]
        result = self.cli(*command)
        self.cached_at = 0
        return result

    def status(self):
        # Signed requests share a bounded 60-second nonce cache. A faster poll
        # can cause 403 and interrupt a good acoustic run. LAN /status is an
        # optional read-only observer, never a credential or audio bridge.
        interval = .2 if self.args.status_url else 1.1
        if self.cached is None or time.monotonic() - self.cached_at >= interval:
            if self.args.status_url:
                with urllib.request.urlopen(self.args.status_url, timeout=6) as response:
                    value = json.load(response)['voice']
            else:
                value = self.proxy('/voice')
            self.cached, self.cached_at = value, time.monotonic()
        return self.cached

    def history(self):
        return self.cli('nyxbot', 'history', self.args.conversation, '--limit', '80')['messages']

    def rows(self):
        return [m for m in self.history() if m['seq'] > self.before and
                (m.get('voice') or {}).get('session_id') == self.session]

    def wait(self, predicate, timeout=70, check_error=True):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.status()
            if predicate(value):
                return value
            if check_error and value['state'] == 'error':
                raise RuntimeError(value['message'])
            time.sleep(.25 if self.args.status_url else 1.1)
        raise TimeoutError('Board state did not converge')

    def play(self, name, background=False):
        path = self.args.fixtures / (name + '.wav')
        if background:
            self.player = subprocess.Popen(['/usr/bin/afplay', str(path)])
        else:
            subprocess.run(['/usr/bin/afplay', str(path)], check=True, timeout=30)

    def reply(self, pattern, after=0):
        deadline = time.monotonic() + 75
        while time.monotonic() < deadline:
            messages = self.rows()
            if any(m['role'] == 'assistant' and m['seq'] > after
                   and (m.get('voice') or {}).get('complete')
                   and (m.get('voice') or {}).get('sealed')
                   and re.search(pattern, m.get('text', ''), re.I) for m in messages):
                return messages
            value = self.status()
            if value['state'] == 'error':
                raise RuntimeError(value['message'])
            time.sleep(1.1)
        raise TimeoutError('No complete expected reply: ' + pattern)

    def quiet(self, user_count):
        self.wait(lambda v: v['message'] != 'Bot speaking' and v['wake']['buffered_ms'] < 100)
        time.sleep(8)
        messages = self.rows()
        assert len([m for m in messages if m['role'] == 'user']) == user_count, messages
        return messages

    def volume(self, user_count):
        after = max((m['seq'] for m in self.rows()), default=self.before)
        expected = self.status()['volume']
        self.play('volume-en')
        self.event('volume_request_finished', expected=expected)
        ones = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
                'seventeen', 'eighteen', 'nineteen']
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        words = (ones[expected] if expected < 20 else 'one hundred' if expected == 100 else
                 tens[expected // 10] + (' ' + ones[expected % 10] if expected % 10 else ''))
        pattern = rf'(?<!\d){expected}(?!\d)|\b' + words.replace(' ', r'[\s-]+') + r'\b'
        self.reply(pattern, after)
        messages = self.quiet(user_count)
        assert any(m['role'] == 'user' and m['seq'] > after and
                   re.search(r'tablet.*current.*volume', m.get('text', ''), re.I) for m in messages), messages
        self.event('volume_reply_complete')

    def full(self):
        self.play('wake-en', background=True)
        value = self.wait(lambda v: v['active'] and v['wake']['initiated_call'])
        self.session = value['session_id']
        self.player.wait(timeout=20)
        self.event('wake_active', session=self.session)
        self.reply(r'purple.*elephant')
        messages = self.quiet(1)
        assert len([m for m in messages if m['role'] == 'assistant']) == 1, messages
        assert any(m['role'] == 'user' and re.search(r'please.*repeat.*exactly.*purple.*elephant',
                   m.get('text', ''), re.I) for m in messages), messages
        self.event('initial_directive_complete_no_echo')
        after = max(m['seq'] for m in messages)
        self.play('count-en')
        self.wait(lambda v: v['message'] == 'Bot speaking', 45)
        time.sleep(.8)
        # Numbered speech can pause between chunks. Start the interruption in
        # an actual playback window, rather than treating a short gap as failure.
        self.wait(lambda v: v['message'] == 'Bot speaking', 45)
        self.event('interrupt_during_playback')
        self.play('interrupt-en')
        self.reply(r'purple.*elephant', after)
        messages = self.quiet(3)
        users = [m for m in messages if m['role'] == 'user' and m['seq'] > after]
        assert any(re.search(r'count.*(?:one|1).*(?:twenty|20)', m.get('text', ''), re.I) for m in users), users
        assert any(re.search(r'stop.*count.*say.*only.*purple.*elephant', m.get('text', ''), re.I) for m in users), users
        self.event('followup_and_interruption_complete_no_echo')
        self.volume(4)
        self.proxy('/voice/microphone', {'muted': True})
        self.wait(lambda v: v['microphone_muted'] and v['wake']['buffered_ms'] == 0)
        self.play('count-en')
        time.sleep(2)
        assert self.status()['wake']['buffered_ms'] == 0
        self.proxy('/voice/microphone', {'muted': False})
        self.wait(lambda v: not v['microphone_muted'])
        self.quiet(4)
        self.event('muted_speech_not_replayed')

    def remote(self):
        self.remote_id = str(uuid.uuid4())
        self.proxy('/voice/open', {'request_id': self.remote_id})
        self.session = self.wait(lambda v: v['active'])['session_id']
        if self.initial['greeting_enabled']:
            self.reply(r'.+')
        messages = self.quiet(0)
        if not self.initial['greeting_enabled']:
            assert not messages, messages
        self.event('remote_open_greeting_policy_no_echo', greeting_enabled=self.initial['greeting_enabled'])
        self.volume(1)

    def boundaries(self):
        before = self.initial['wake']['detections']
        self.proxy('/voice/wake', {'enabled': False})
        self.wait(lambda v: not v['wake']['listening'])
        self.play('wake-en')
        time.sleep(2)
        value = self.status()
        assert not value['busy'] and value['wake']['detections'] == before, value
        self.event('disabled_wake_does_not_start_call')
        self.proxy('/voice/wake', {'enabled': True})
        self.wait(lambda v: v['wake']['listening'])
        self.play('wake-en', background=True)
        value = self.wait(lambda v: v['busy'] and v['wake']['initiated_call'] and
                          v['wake']['buffered_ms'] > 0)
        self.session = value.get('session_id') or None
        assert not value['active'], 'Connected before cancellation; connecting cancellation was not tested'
        self.proxy('/voice/stop', {})
        self.player.wait(timeout=20)
        value = self.wait(lambda v: not v['busy'] and v['wake']['listening'])
        assert value['wake']['buffered_ms'] == 0 and not value['active'], value
        self.event('cancel_connecting_discards_recording_and_rearms')
        self.proxy('/voice/stop', {})
        assert not self.wait(lambda v: not v['busy'])['active']
        self.event('repeated_hangup_safe')

    def cleanup(self):
        errors = []
        try:
            if self.player and self.player.poll() is None:
                self.player.terminate()
                self.player.wait(timeout=10)
        except Exception as error:
            errors.append(f'player: {error}')
        # Losing history access must not skip hangup or restoration.
        try:
            if self.session:
                self.report['messages'] = [{k: m.get(k) for k in ['seq', 'role', 'text', 'voice']}
                                           for m in self.rows()]
        except Exception as error:
            errors.append(f'history: {error}')
        try:
            value = self.status()
            owns = ((self.session and value.get('session_id') == self.session)
                    or (self.remote_id and value.get('remote_request_id') == self.remote_id)
                    or (self.args.scenario != 'remote' and value['wake']['initiated_call']
                        and value['wake']['detections'] > self.initial['wake']['detections']))
            if owns and value['busy']:
                try:
                    if value['microphone_muted']:
                        self.proxy('/voice/microphone', {'muted': self.initial['microphone_muted']})
                finally:
                    self.proxy('/voice/stop', {})
                self.wait(lambda v: not v['busy'], check_error=False)
        except Exception as error:
            errors.append(f'hangup: {error}')
        try:
            if self.args.scenario == 'boundaries':
                self.proxy('/voice/wake', {'enabled': self.initial['wake']['enabled']})
        except Exception as error:
            errors.append(f'wake restore: {error}')
        try:
            self.report['final_voice'] = self.status()
        except Exception as error:
            errors.append(f'final status: {error}')
        if errors:
            self.report['cleanup_error'] = errors
            self.report['passed'] = False

    def run(self):
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        self.initial = self.wait(lambda v: not v['busy'], check_error=False)
        assert not self.initial['microphone_muted'], 'Unmute before an acoustic test'
        if self.args.scenario != 'remote':
            assert self.initial['wake']['enabled'] and self.initial['wake']['listening'], self.initial
        self.before = max((m['seq'] for m in self.history()), default=0)
        try:
            getattr(self, self.args.scenario)()
            self.report['passed'] = True
        except BaseException as error:
            self.report['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            self.cleanup()
            self.args.output.write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + '\n')
            self.args.output.chmod(0o600)
        if not self.report.get('passed'):
            raise RuntimeError('Acceptance or cleanup failed; see report')
        print('PASS', self.args.scenario, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', required=True, help='Daily service ID or slug')
    parser.add_argument('--conversation', required=True)
    parser.add_argument('--profile', help='Saved nyxid CLI owner profile; never a provider or board Key')
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--status-url', help='Optional read-only LAN URL, e.g. http://panel.local/status')
    parser.add_argument('--scenario', choices=['full', 'remote', 'boundaries'], default='full')
    args = parser.parse_args()
    required = {'full': ['wake-en', 'count-en', 'interrupt-en', 'volume-en'],
                'remote': ['volume-en'], 'boundaries': ['wake-en']}[args.scenario]
    for name in required:
        if not (args.fixtures / (name + '.wav')).is_file():
            parser.error('Missing fixture: ' + name + '.wav')
    Acceptance(args).run()


if __name__ == '__main__':
    main()
