import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline

# CONFIG
DATA_PATH = r"D:\datasets\calvin_debug\test"
DEGREE    = 3
FPS       = 30

#fix the window size to 30 for the sample's rate, but you can change it to any value you want

WINDOWS_TO_TEST = [30]

# number of segments to sample per window size, you can change it to any value you want
SEGMENTS_PER_WINDOW = 10

#spread or random, spread will sample segments evenly across the dataset, while random will sample segments randomly, need seed to recreate
SAMPLING_STRATEGY = "spread"
SEED = 0

MAX_SEGMENTS_TO_PLOT = 2
NUM_CTRL_PTS_PER_DIM = [12, 12, 12, 0, 16, 14, 0]
SKIP_DIMS    = {3, 6}
ANGULAR_DIMS = {5}

names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

# if there is an name that is either official or something that you want to start with for the boundary file, you can load it here, otherwise leave it as None
# EP_START_END_IDS = np.load(os.path.join(DATA_PATH, "ep_start_end_ids.npy"))
EP_START_END_IDS = None
# HELPERS
def make_knots(n_ctrl, degree):
    return np.concatenate([
        np.zeros(degree),
        np.linspace(0, 1, n_ctrl - degree + 1),
        np.ones(degree)
    ])


def extract_frame_id(filename):
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else None


def is_valid_window(all_files, start, window, ep_start_end_ids=None):
    chunk = all_files[start:start + window]
    ids = [extract_frame_id(f) for f in chunk]

    if ep_start_end_ids is not None:
        if any(i is None for i in ids):
            return False
        s0, e0 = ids[0], ids[-1]
        return any(s <= s0 and e0 <= e for s, e in ep_start_end_ids)

    if any(i is None for i in ids):
        return True  # if there exists a file that doesn't have a frame id, we will not check for continuity
    return all(ids[i + 1] - ids[i] == 1 for i in range(len(ids) - 1))


def find_contiguous_window(all_files, window, ep_start_end_ids=None, start_from=0):
    n = len(all_files)
    if n < window:
        raise ValueError(f"文件总数 {n} 小于 window {window}")

    for start in range(start_from, n - window + 1):
        if is_valid_window(all_files, start, window, ep_start_end_ids):
            return start, all_files[start:start + window]

    raise RuntimeError(
        f"从 index {start_from} 开始,找不到长度为 {window} 且不跨 episode 边界的连续窗口"
    )


def collect_segments(all_files, window, num_segments, ep_start_end_ids=None,
                      strategy="spread", seed=0):
    #find multiple non-overlapping segments of length `window` from `all_files`. strategy: "spread" or "random" tho random will sample segments randomly, need seed to recreate, and you might face an issue that you can't find enough segments, so you might want to use spread instead
    n = len(all_files)
    if n < window:
        raise ValueError(f"文件总数 {n} 小于 window {window}")

    max_start = n - window
    segments = []  # list of (start_idx, chunk_files)
    used_ranges = []

    def overlaps_used(start):
        return any(not (start + window <= s or start >= e) for s, e in used_ranges)

    if strategy == "spread":
        bounds = np.linspace(0, max_start, num_segments + 1).astype(int)
        for i in range(num_segments):
            lo, hi = bounds[i], bounds[i + 1]
            found = None
            for start in range(lo, hi + 1):
                if start > max_start:
                    break
                if overlaps_used(start) or not is_valid_window(all_files, start, window, ep_start_end_ids):
                    continue
                found = start
                break
            if found is not None:
                segments.append((found, all_files[found:found + window]))
                used_ranges.append((found, found + window))

    elif strategy == "random":
        rng = np.random.RandomState(seed)
        candidates = list(range(0, max_start + 1))
        rng.shuffle(candidates)
        for start in candidates:
            if len(segments) >= num_segments:
                break
            if overlaps_used(start) or not is_valid_window(all_files, start, window, ep_start_end_ids):
                continue
            segments.append((start, all_files[start:start + window]))
            used_ranges.append((start, start + window))

    else:
        raise ValueError(f"unknown strategy: {strategy}")

    if len(segments) < num_segments:
        print(f"  warning: there exists only {len(segments)}/{num_segments} segments of length {window} that do not cross episode boundaries(could be due to the dataset itself or the sampling strategy).")

    return segments

# FIT SPLINES
def fit_splines(chunk_actions, timestamps):
    L, D = chunk_actions.shape

    t = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])

    actions_fit = chunk_actions.copy()
    for d in ANGULAR_DIMS:
        actions_fit[:, d] = np.unwrap(actions_fit[:, d])

    ctrl_pts = []
    knots_list = []

    for d in range(D):
        if d in SKIP_DIMS:
            ctrl_pts.append(chunk_actions[:, d])
            knots_list.append(None)
            continue

        n_ctrl = NUM_CTRL_PTS_PER_DIM[d]
        knots = make_knots(n_ctrl, DEGREE)
        spl = make_lsq_spline(t, actions_fit[:, d], knots, k=DEGREE)

        ctrl_pts.append(spl.c.copy())
        knots_list.append(knots)

    return ctrl_pts, knots_list


# RECONSTRUCT
def reconstruct(ctrl_pts, knots_list, timestamps):
    L = len(timestamps)
    t = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])

    result = np.zeros((L, len(ctrl_pts)))

    for d, cp in enumerate(ctrl_pts):
        if d in SKIP_DIMS:
            result[:, d] = cp
            continue

        knots = knots_list[d]
        vals = BSpline(knots, cp, DEGREE)(t)

        if d in ANGULAR_DIMS:
            vals = (vals + np.pi) % (2 * np.pi) - np.pi

        result[:, d] = vals

    return result


# RUN ONE SEGMENT (single fit + reconstruct + MSE, no plotting)
def run_one_segment(chunk_files, window):
    actions = np.array([
        np.load(os.path.join(DATA_PATH, f))["actions"]
        for f in chunk_files
    ])
    timestamps = np.arange(window) / FPS

    ctrl_pts, knots_list = fit_splines(actions, timestamps)
    reconstructed = reconstruct(ctrl_pts, knots_list, timestamps)

    mse_list = [
        np.mean((actions[:, d] - reconstructed[:, d]) ** 2)
        for d in range(7)
    ]
    return actions, reconstructed, timestamps, mse_list


def plot_segment(actions, reconstructed, timestamps, mse_list, window, start_idx, seg_idx):
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for d, (name, ax) in enumerate(zip(names, axes)):
        ax.plot(timestamps, actions[:, d], label="Original", alpha=0.8)
        ax.plot(timestamps, reconstructed[:, d], '--', label="Spline", alpha=0.9)

        tag = "[raw]" if d in SKIP_DIMS else f"MSE: {mse_list[d]:.6f}"
        ax.set_title(f"{name} ({tag})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    axes[-1].set_visible(False)
    plt.suptitle(
        f"WINDOW={window}, segment #{seg_idx} (files[{start_idx}:{start_idx + window}])",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


# RUN ONE WINDOW SIZE, MULTIPLE SEGMENTS
def run_for_window(all_files, window):
    print(f"\n{'=' * 50}")
    print(f"WINDOW = {window}  |  SEGMENTS = {SEGMENTS_PER_WINDOW}  |  strategy = {SAMPLING_STRATEGY}")
    print(f"{'=' * 50}")

    segments = collect_segments(
        all_files, window, SEGMENTS_PER_WINDOW,
        ep_start_end_ids=EP_START_END_IDS,
        strategy=SAMPLING_STRATEGY,
        seed=SEED,
    )
    per_segment_mse = []  # list of length-7 mse lists
    for seg_idx, (start_idx, chunk_files) in enumerate(segments):
        actions, reconstructed, timestamps, mse_list = run_one_segment(chunk_files, window)
        per_segment_mse.append(mse_list)

        avg_mse = np.mean([mse_list[d] for d in range(7) if d not in SKIP_DIMS])
        print(f"  segment #{seg_idx} (start={start_idx:>6d}): avg MSE (fitted dims) = {avg_mse:.8f}")

        if seg_idx < MAX_SEGMENTS_TO_PLOT:
            plot_segment(actions, reconstructed, timestamps, mse_list, window, start_idx, seg_idx)

    per_segment_mse = np.array(per_segment_mse)  # (num_segments, 7)

    print(f"\nMSE Statistics (across {len(segments)} segments):")
    print(f"{'dim':10s} {'mean':>12s} {'std':>12s} {'min':>12s} {'max':>12s}")
    for d, name in enumerate(names):
        col = per_segment_mse[:, d]
        tag = " [raw]" if d in SKIP_DIMS else ""
        print(f"{name:10s} {col.mean():12.8f} {col.std():12.8f} {col.min():12.8f} {col.max():12.8f}{tag}")

    fitted_dims = [d for d in range(7) if d not in SKIP_DIMS]
    avg_per_segment = per_segment_mse[:, fitted_dims].mean(axis=1)
    print(f"\nAvg MSE (fitted dims) across segments: "
          f"mean={avg_per_segment.mean():.8f}  std={avg_per_segment.std():.8f}")

    return per_segment_mse



# MAIN

if __name__ == "__main__":
    all_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".npz")])

    results = {}
    for w in WINDOWS_TO_TEST:
        results[w] = run_for_window(all_files, w)

    print(f"\n{'=' * 50}")
    print("comparison (Avg MSE across segments, fitted dims only)")
    print(f"{'=' * 50}")
    fitted_dims = [d for d in range(7) if d not in SKIP_DIMS]
    for w in WINDOWS_TO_TEST:
        per_segment_mse = results[w]  # (num_segments, 7)
        avg_per_segment = per_segment_mse[:, fitted_dims].mean(axis=1)
        print(f"WINDOW={w:4d}: mean={avg_per_segment.mean():.8f}  "
              f"std={avg_per_segment.std():.8f}  n_segments={len(avg_per_segment)}")

