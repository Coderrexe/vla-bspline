"""Offline scene-coverage audit from archived telemetry; never connects to hardware.

Builds only a fresh MuJoCo scene and runs forward kinematics. It does not step
physics, construct a workcell, publish actions, or change a live scene. TCP
projection is a coverage diagnostic, NOT fingertip clearance certification.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--telemetry', type=Path, required=True)
    p.add_argument('--session-id', required=True)
    p.add_argument('--scene', default='mavis_v2')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new audit output')

    # Simulation package only. In particular, no runtime or hardware imports.
    import mujoco
    from apollo_mavis_v2_sim import REGISTRY, SceneOverrides, asset_path

    scene = REGISTRY.build(args.scene, SceneOverrides(microphones={'view': True}))
    model = scene.model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    table_id = int(model.geom('table').id)
    if model.geom_type[table_id] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError('This audit requires a box named table')
    table_halfsize = model.geom_size[table_id].copy()
    rows = []
    with args.telemetry.open() as f:
        for line in f:
            frame = json.loads(line)
            session = frame.get('session') or {}
            if session.get('session_id') != args.session_id or session.get('state') != 'running':
                continue
            arms = {a['arm_id']: a for a in frame.get('arms', [])}
            if set(arms) != set(scene.addressing.arms):
                raise ValueError('Expected both arms in archived running telemetry')
            for aid, address in scene.addressing.arms.items():
                q = np.asarray(arms[aid]['q'], dtype=float)
                if address.has_rail:
                    q = np.r_[q, float(arms[aid]['rail_pos_m'])]
                if q.shape != address.qpos_adr.shape or not np.isfinite(q).all():
                    raise ValueError('Invalid archived joint vector')
                data.qpos[address.qpos_adr] = q
            mujoco.mj_kinematics(model, data)
            address = scene.addressing['grip']
            base_rotation = data.xmat[address.base_body_id].reshape(3, 3)
            tcp_base = np.asarray(arms['grip']['ee_pose']['position'], dtype=float)
            if tcp_base.shape != (3,) or not np.isfinite(tcp_base).all():
                raise ValueError('Invalid archived TCP')
            tcp_world = data.xpos[address.base_body_id] + base_rotation @ tcp_base
            fk_tcp_world = data.site_xpos[address.tcp_site_id].copy()
            reported_rotation = Rotation.from_matrix(base_rotation)*Rotation.from_quat(
                np.asarray(arms['grip']['ee_pose']['orientation'])[[1,2,3,0]])
            fk_rotation = Rotation.from_matrix(data.site_xmat[address.tcp_site_id].reshape(3,3))
            table_rotation = data.geom_xmat[table_id].reshape(3, 3)
            table_local = table_rotation.T @ (tcp_world - data.geom_xpos[table_id])
            excess = np.maximum(np.abs(table_local[:2]) - table_halfsize[:2], 0.)
            rows.append({
                'ts': frame['ts'], 'tcp_base_m': tcp_base.tolist(),
                'tcp_world_m': tcp_world.tolist(),
                'fk_tcp_world_m': fk_tcp_world.tolist(),
                'tcp_fk_disagreement_mm': float(np.linalg.norm(tcp_world-fk_tcp_world)*1000),
                'orientation_fk_disagreement_deg': float(np.degrees(
                    (reported_rotation*fk_rotation.inv()).magnitude())),
                'tcp_projection_outside_table_mm': float(np.linalg.norm(excess)*1000),
                'tcp_height_above_model_table_plane_mm': float((table_local[2]-table_halfsize[2])*1000),
                'gripper_open_frac': arms['grip']['gripper_open_frac'],
            })
    if not rows:
        raise ValueError('No matching running telemetry')
    scene_path = asset_path(f'scenes/{args.scene}.yaml')
    report = {
        'purpose': 'OFFLINE_TABLE_COVERAGE_NOT_PHYSICAL_CLEARANCE_CERTIFICATION',
        'hardware_connection': False, 'physics_steps': 0,
        'session_id': args.session_id, 'scene': args.scene,
        'telemetry_sha256': hashlib.sha256(args.telemetry.read_bytes()).hexdigest(),
        'scene_yaml_sha256': hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        'compiled_xml_sha256': hashlib.sha256(scene.xml.encode()).hexdigest(),
        'scene_yaml': str(scene_path),
        'table_center_world_m': data.geom_xpos[table_id].tolist(),
        'table_halfsize_m': table_halfsize.tolist(),
        'frames': len(rows),
        'max_tcp_fk_disagreement_mm': max(r['tcp_fk_disagreement_mm'] for r in rows),
        'max_orientation_fk_disagreement_deg': max(r['orientation_fk_disagreement_deg'] for r in rows),
        'first': rows[0], 'last': rows[-1],
        'minimum_tcp_height': min(rows, key=lambda r: r['tcp_height_above_model_table_plane_mm']),
        'limitations': [
            'Model dimensions are not freshly measured lab geometry.',
            'TCP is not the lowest fingertip; no physical clearance is certified.',
            'The operator contact time is unknown; last pose is not a calibrated contact plane.',
            'No production scene, runtime setting, or controller was changed.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
