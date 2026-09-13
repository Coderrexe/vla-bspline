"""Bounded read-only capture of Apollo's existing real wrist-video streams.

No SDK, action output, session creation, or runtime change. Saves the existing
JPEG packets and timestamp/sequence headers without decoding or re-encoding.
"""
import argparse
import asyncio
import json
from pathlib import Path
import struct
import time

import websockets


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=30)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error('Capture must be bounded to at most 60 seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    ready = set()

    async def capture(camera):
        folder = args.output/camera
        folder.mkdir()
        count = 0
        async with websockets.connect(
                f'ws://127.0.0.1:8765/ws/video/{camera}', open_timeout=5) as ws:
            deadline = time.monotonic()+args.seconds
            with (folder/'frames.jsonl').open('w') as index:
                while time.monotonic() < deadline:
                    packet = await asyncio.wait_for(ws.recv(), timeout=5)
                    if not isinstance(packet, bytes) or packet[12:14] != b'\xff\xd8':
                        raise ValueError('Unexpected Apollo JPEG wire format')
                    timestamp, sequence = struct.unpack('<dI', packet[:12])
                    filename = f'{count:06d}.jpg'
                    (folder/filename).write_bytes(packet[12:])
                    index.write(json.dumps({'file': filename, 'timestamp': timestamp,
                                            'sequence': sequence,
                                            'received_t_mono': time.monotonic()})+'\n')
                    count += 1
                    if camera not in ready:
                        ready.add(camera)
                        if len(ready) == 2:
                            print('Both real camera streams are recording; no robot commands', flush=True)
        return {'camera': camera, 'frames': count}

    results = await asyncio.gather(capture('view_wrist'), capture('grip_wrist'))
    (args.output/'report.json').write_text(json.dumps({
        'mode': 'READ_ONLY_VIDEO_CAPTURE', 'runtime_api_writes': 0,
        'action_messages_sent': 0, 'streams': results}, indent=2)+'\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
