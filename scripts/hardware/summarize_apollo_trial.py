"""Summarize archived session telemetry and learned commands; no robot interface."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--video',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    check=json.loads((args.session/'session_check.json').read_text())
    report={'purpose':'ARCHIVED_COMMISSIONING_MEASUREMENT_NOT_TASK_SUCCESS',
            'session_id':check.get('created',{}).get('session_id'),
            'helper_status':check['status'],'sources':{},'task_success_scored':False}
    paths=[args.session/'session_check.json',args.session/'telemetry.jsonl',
           args.policy/'predictions.jsonl']
    for path in paths:
        if path.exists():report['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    messages=[json.loads(x) for x in paths[1].read_text().splitlines()] if paths[1].exists() else []
    running=[x for x in messages if x.get('session',{}).get('state')=='running']
    predictions=[json.loads(x) for x in paths[2].read_text().splitlines()] if paths[2].exists() else []
    published=[x for x in predictions if x['publication_requested']]
    report.update(running_telemetry_frames=len(running),published_chunks=len(published),
                  reported_fault_frames=sum(bool(x.get('collision',{}).get('blocked')) or
                    any(y.get('error_code') or y.get('stale') or y.get('recovering')
                        for y in x.get('arms',[])) for x in running))
    if published:
        actions=np.concatenate([np.asarray(x['actions_grip'])[:x['execution_rows']] for x in published])
        report.update(published_rows=len(actions),
                      requested_translation_sum_mm=(actions[:,:3].sum(0)*1000).tolist(),
                      requested_translation_path_mm=float(np.linalg.norm(actions[:,:3],axis=1).sum()*1000),
                      requested_final_gripper=float(actions[-1,6]),
                      compute_ms=[x['compute_ms'] for x in published])
    first=next((i for i,x in enumerate(running) if x.get('external',{}).get('action_age_s') is not None),None)
    if first is not None and first>0:
        before,after=running[first-1],running[-1]
        report.update(before_action_t_mono=before['ts'],after_action_t_mono=after['ts'],arms={})
        for name in ('grip','view'):
            a,z=[next(y for y in x['arms'] if y['arm_id']==name) for x in (before,after)]
            delta=np.asarray(z['ee_pose']['position'])-a['ee_pose']['position']
            report['arms'][name]={'measured_tcp_delta_mm':(delta*1000).tolist(),
                'measured_tcp_displacement_mm':float(np.linalg.norm(delta)*1000),
                'max_joint_change_rad':float(np.max(np.abs(np.asarray(z['q'])-a['q']))),
                'rail_delta_m':z['rail_pos_m']-a['rail_pos_m'],
                'gripper_before':a['gripper_open_frac'],'gripper_after':z['gripper_open_frac']}
        if args.video:
            from PIL import Image,ImageDraw
            targets=np.linspace(before['ts'],after['ts'],4).tolist()
            canvas=Image.new('RGB',(1280,520),'white')
            for row,camera in enumerate(('view_wrist','grip_wrist')):
                folder=args.video/camera
                frames=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()]
                for col,target in enumerate(targets):
                    frame=min(frames,key=lambda x:abs(x['timestamp']-target))
                    if abs(frame['timestamp']-target)>.1:
                        raise ValueError('Video does not cover requested telemetry timestamp')
                    with Image.open(folder/frame['file']) as im:
                        canvas.paste(im.resize((320,240)),(col*320,row*260+20))
                    ImageDraw.Draw(canvas).text((col*320+5,row*260+4),
                        f'{camera} {frame["timestamp"]-targets[0]:+.2f}s',fill='black')
            canvas.save(args.output/'action_response_frames.png')
    with (args.output/'report.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
