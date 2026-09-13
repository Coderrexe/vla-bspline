"""Bounded read-only Dora command/state logger; observer has no outputs."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request
from collections import Counter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=30)
    args=p.parse_args()
    if not 0 < args.seconds <= 60: p.error('Use a bounded capture of at most 60 seconds')
    args.output.mkdir(parents=True,exist_ok=False)
    with urllib.request.urlopen('http://127.0.0.1:8765/api/dora',timeout=5) as response:
        info=json.load(response)
    if info['state']!='attached' or 'observer' not in info['placeholders']:
        raise RuntimeError('No existing observer placeholder')
    os.environ.update(DORA_ZENOH_CONNECT=info['zenoh_connect'],DORA_ZENOH_LISTEN='tcp/127.0.0.1:0',
                      DORA_ZENOH_MULTICAST='off')
    from dora import Node
    node=Node('observer',daemon_port=info['daemon_port'])
    count=Counter();start=time.monotonic()
    print('Read-only command/state observer attached',flush=True)
    try:
        with (args.output/'commands_states.jsonl').open('x') as stream:
            while time.monotonic()-start < args.seconds:
                event=node.next(timeout=.2)
                if event is None or event.get('type') in ('STOP','INPUT_CLOSED'):break
                if event.get('type')!='INPUT':continue
                topic=event['id']
                if topic not in {'arm_cmd','arm_state','session','events'}:continue
                count[topic]+=1
                stream.write(json.dumps({'topic':topic,'metadata':dict(event.get('metadata') or {}),
                                         'values':event['value'].to_pylist()},default=str)+'\n')
    finally:
        del node
        (args.output/'report.json').write_text(json.dumps({'mode':'READ_ONLY_COMMAND_OBSERVER',
            'counts':dict(count),'duration_s':time.monotonic()-start,'actions_published':0,
            'runtime_api_writes':0},indent=2)+'\n')


if __name__=='__main__':main()
