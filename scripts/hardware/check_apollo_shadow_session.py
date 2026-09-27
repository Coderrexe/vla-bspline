"""One explicitly authorized, bounded hardware session for a SHADOW client.

Uses only the documented HTTP/telemetry API. No action publication or direct
hardware calls. Creates a keep-current 10% session, records telemetry, and closes
only that session through DELETE (never the UI's return-home operation).
Run ONLY with the operator's approval and a separately verified shadow client.
"""
import argparse
import asyncio
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import websockets
from apollo_contact_review import require_contact_review_clear


def request(url, method='GET', body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=40) as response:
        content = response.read()
        return json.loads(content) if content else None


def current_session(url):
    try:
        return request(url+'/api/session')
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirm-supervised-shadow',action='store_true')
    p.add_argument('--expected-policy-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--url',default='http://127.0.0.1:8765')
    args = p.parse_args()
    # Enabling a shadow hardware session can move the arms during startup.
    require_contact_review_clear()
    if not args.confirm_supervised_shadow or not 0 < args.seconds <= 60:
        p.error('Requires explicit supervised-shadow authorization and 1–60 s duration')
    args.output.mkdir(parents=True,exist_ok=False)
    report = {'purpose':'SUPERVISED_SHADOW_ONLY','action_publication':False,
              'session_create_posts':0,'session_delete_requests':0,
              'expected_policy_id':args.expected_policy_id}
    own_session = None
    session_url = args.url+'/api/session'
    try:
        if current_session(args.url) is not None:
            raise RuntimeError('An existing session is present; do not replace it')
        wc = request(args.url+'/api/workcell?kind=hardware')
        if not wc.get('hardware_ready') or not wc.get('policy_modes'):
            raise RuntimeError('Hardware is not ready for an external policy session')
        if any(a['error_code'] or a['connected'] or a['reachable'] != 'open' for a in wc['arms']):
            raise RuntimeError('A controller is not healthy and idle')
        ws_url = args.url.replace('http://','ws://').replace('https://','wss://')+'/ws/telemetry'
        async with websockets.connect(ws_url,open_timeout=5) as ws:
            before = json.loads(await asyncio.wait_for(ws.recv(),timeout=5))
            report['before'] = before
            ext = before.get('external') or {}
            if (not ext.get('policy_attached') or ext.get('policy_id') != args.expected_policy_id
                    or ext.get('policy_arms') != ['grip'] or ext.get('spec_age_s',99) > 1.5):
                raise RuntimeError('The expected fresh grip-only shadow policy is not attached')
            # Validate body explicitly; omitted return_to_start is inert in inference.
            body = {'mode':'inference','kind':'hardware','arms':['grip','view'],
                    'frames':{'grip':'arm_base:grip','view':'arm_base:view'},
                    'digital_twin_scene':'mavis_v2','start_from':'keep_current',
                    'speed_scale':.1,'policy_source':'external','policy':None,
                    'task':'Drawer Assembling'}
            report['request'] = body
            if current_session(args.url) is not None:
                raise RuntimeError('A session appeared during preflight; leaving it untouched')
            report['session_create_posts'] += 1
            print('Creating approved shadow session; keep_current; speed_scale=0.1; no policy actions',flush=True)
            try:
                created = await asyncio.to_thread(request, session_url, 'POST', body)
            except urllib.error.HTTPError as exc:
                report['creation_http_status'] = exc.code
                report['creation_error_body'] = exc.read().decode()
                raise
            own_session = created['session_id']
            report['created'] = created
            print(json.dumps({'created':created}),flush=True)
            deadline = time.monotonic()+args.seconds
            rows = 0
            with (args.output/'telemetry.jsonl').open('w') as output:
                while time.monotonic() < deadline:
                    message = json.loads(await asyncio.wait_for(ws.recv(),timeout=5))
                    s = message.get('session') or {}
                    if s.get('session_id') != own_session:
                        continue  # drain pre-creation telemetry without interpreting it
                    output.write(json.dumps(message)+'\n')
                    rows += 1
                    report['last'] = message
                    ext = message.get('external') or {}
                    if ext.get('action_age_s') is not None:
                        raise RuntimeError('Unexpected action receipt in shadow session; ending our session')
                    if s.get('state') in {'fault','recovering','teardown'}:
                        report['status'] = 'SESSION_NOT_RUNNING'
                        break
            report['telemetry_frames'] = rows
            report.setdefault('status','OBSERVED_SHADOW' if rows else 'NO_SESSION_TELEMETRY')
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report.setdefault('status','REFUSED_OR_ERROR')
        raise
    finally:
        try:
            if own_session:
                active = current_session(args.url)
                if active and active['session_id'] == own_session:
                    request(session_url,'DELETE')
                    report['session_delete_requests'] += 1
                    print('Closed our shadow session without return-home or policy action',flush=True)
                else:
                    report['cleanup_note'] = 'Our session is no longer active; no DELETE sent'
                report['session_after_cleanup'] = current_session(args.url)
        finally:
            (args.output/'session_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'before','last'}},indent=2),flush=True)
    return 0 if report['status'] == 'OBSERVED_SHADOW' else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
