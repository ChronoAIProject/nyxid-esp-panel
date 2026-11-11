"""Serve the pilot and save explicit submissions in a private local folder."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path

from prepare_review import PROTOCOL, session, sha

MIME = {'/': ('index.html', 'text/html; charset=utf-8'),
        '/capture.js': ('capture.js', 'text/javascript; charset=utf-8'),
        '/recorder.js': ('recorder.js', 'text/javascript; charset=utf-8'),
        '/protocol.json': ('protocol.json', 'application/json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8796)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--test-fixtures', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.output.stat().st_mode & 0o077:
        parser.error('Capture directory must be private (mode 700)')
    spec = json.loads(PROTOCOL.read_bytes())
    host = f'127.0.0.1:{args.port}'

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, content_type, body):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get('Host') != host or self.path not in MIME:
                return self.reply(404, 'text/plain', b'Not found')
            name, kind = MIME[self.path]
            self.reply(200, kind, PROTOCOL.with_name(name).read_bytes())

        def do_POST(self):
            if (self.path != '/save' or self.headers.get('Host') != host
                    or self.headers.get('Origin') != f'http://{host}'):
                return self.reply(403, 'text/plain', b'Local recording page only')
            self.connection.settimeout(5)
            try:
                size = int(self.headers['Content-Length'])
                if not 0 < size <= 9_000_000 or self.headers.get_content_type() != 'application/json':
                    raise ValueError('Unexpected recording format')
                blob = self.rfile.read(size)
                if len(blob) != size:
                    raise ValueError('Incomplete submission')
                who, condition, _ = session(blob, spec, args.test_fixtures)
            except (KeyError, ValueError, TypeError, OSError):
                return self.reply(400, 'text/plain', b'Recording failed local validation')
            name = f'nyx-wake-{who}-{condition}-{sha(blob)[:12]}.json'
            try:
                with (args.output / name).open('xb') as handle:
                    handle.write(blob)
            except FileExistsError:
                if (args.output / name).read_bytes() != blob:
                    return self.reply(409, 'text/plain', b'Conflicting local recording')
            except OSError:
                return self.reply(500, 'text/plain', b'Could not save the local recording')
            self.reply(200, 'text/plain', b'Saved to the local research folder.')

    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(f'Recording page: http://{host}/', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
