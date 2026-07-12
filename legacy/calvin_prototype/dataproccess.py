
import os
import re
import numpy as np
from scipy.interpolate import make_lsq_spline
from tqdm import tqdm

# CONFIG
DATA_PATH   = r"D:\datasets\calvin_debug\test"
SAVE_NAME   = "calvin_vla_dataset.npz"
WINDOW      = 30
DEGREE      = 3

DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
NUM_CTRL_PTS_PER_DIM = [12, 12, 12, 0, 16, 14, 0]
SKIP_DIMS    = {3, 6}   # roll, gripper -> raw
ANGULAR_DIMS = {5}      # yaw -> unwrap

# valid_dims: 1 = spline, 0 = raw
VALID_DIMS = np.array([0 if d in SKIP_DIMS else 1 for d in range(7)], dtype=np.int64)

# ep_start_end_ids = np.load(os.path.join(DATA_PATH, "ep_start_end_ids.npy"))
EP_START_END_IDS = None  
# HELPERS
def make_knots(n_ctrl, degree):
    """Clamped uniform knot vector, length = n_ctrl + degree + 1."""
    return np.concatenate([
        np.zeros(degree),
        np.linspace(0, 1, n_ctrl - degree + 1),
        np.ones(degree),
    ])


def extract_frame_id(filename):
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else None

def chunk_is_contiguous(chunk_files):
    ids = [extract_frame_id(f) for f in chunk_files]
    if any(i is None for i in ids):
        return True 
    return all(ids[i + 1] - ids[i] == 1 for i in range(len(ids) - 1))

def chunk_crosses_episode_boundary(start_idx, end_idx, ep_start_end_ids):
    if ep_start_end_ids is None:
        return False
    for s, e in ep_start_end_ids:
        if s <= start_idx <= e:
            return not (s <= end_idx - 1 <= e)
    return True 
# SPLINE FIT
def fit_splines(chunk_actions):
    L, D = chunk_actions.shape
    t = np.linspace(0, 1, L)

    actions_fit = chunk_actions.copy()
    for d in ANGULAR_DIMS:
        actions_fit[:, d] = np.unwrap(actions_fit[:, d])

    ctrl_pts = []
    for d in range(D):
        if d in SKIP_DIMS:
            ctrl_pts.append(chunk_actions[:, d].copy())
            continue

        n_ctrl = NUM_CTRL_PTS_PER_DIM[d]
        knots = make_knots(n_ctrl, DEGREE)

        spl = make_lsq_spline(t, actions_fit[:, d], knots, k=DEGREE)
        ctrl_pts.append(spl.c.copy())

    return ctrl_pts
# LOAD + PROCESS
def build_dataset():
    files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".npz")])
    step = WINDOW

    images_all, ctrl_all, start_all, traj_len_all = [], [], [], []
    n_skipped_boundary = 0

    print(f"Total files: {len(files)}")
    n_chunks_expected = (len(files) - step) // step + 1 if len(files) >= step else 0
    print(f"Expected chunks: {n_chunks_expected}")
    for i in tqdm(range(0, len(files) - step + 1, step)):
        chunk_files = files[i:i + step]
        if EP_START_END_IDS is not None:
            crosses = chunk_crosses_episode_boundary(i, i + step, EP_START_END_IDS)
        else:
            crosses = not chunk_is_contiguous(chunk_files)

        if crosses:
            n_skipped_boundary += 1
            continue
        rgb_list, action_list, robot_obs_list = [], [], []
        for f in chunk_files:
            with np.load(os.path.join(DATA_PATH, f)) as data:
                rgb_list.append(data["rgb_static"])
                action_list.append(data["actions"])
                robot_obs_list.append(data["robot_obs"] if "robot_obs" in data else data["actions"])

        chunk_imgs = np.array(rgb_list)
        chunk_actions = np.array(action_list)
        chunk_robot_obs = np.array(robot_obs_list)

        ctrl = fit_splines(chunk_actions)

        images_all.append(chunk_imgs)
        ctrl_all.append(ctrl)
        start_all.append(chunk_robot_obs[0])   
        traj_len_all.append(len(chunk_actions))

    if n_skipped_boundary:
        print(f"Skip {n_skipped_boundary} numbers of episode boundary chunks, to avoid meaningless trajectories")

    return images_all, ctrl_all, start_all, traj_len_all
# SAVE
def main():
    images_all, ctrl_all, start_all, traj_len_all = build_dataset()

    ctrl_arr = np.array(ctrl_all, dtype=object)

    np.savez(
        SAVE_NAME,
        images          = np.array(images_all),  
        control_points  = ctrl_arr,                   
        valid_dims      = VALID_DIMS,
        raw_start_state = np.array(start_all),
        traj_lengths    = np.array(traj_len_all),
        dim_names       = np.array(DIM_NAMES),
        num_ctrl_pts_per_dim = np.array(NUM_CTRL_PTS_PER_DIM),
        degree          = DEGREE,
    )
    print("\nDONE")
    print("images shape:", np.array(images_all).shape)
    print("num chunks  :", len(ctrl_all))
    print("valid_dims  :", dict(zip(DIM_NAMES, VALID_DIMS.tolist())))
    print("\n加载时记得: np.load(SAVE_NAME, allow_pickle=True)  # object array 需要 allow_pickle")


if __name__ == "__main__":
    main()

