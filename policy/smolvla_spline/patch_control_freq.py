"""Idempotently thread `control_freq` through lerobot's LIBERO env stack, so
`--env.control_freq=10/20/40` works for Hz-mismatch experiments.

Run: python patch_control_freq.py /path/to/lerobot/src/lerobot
"""
import os
import sys

EDITS = {
    "envs/configs.py": [
        (
            '    control_mode: str = "relative"  # or "absolute"\n',
            '    control_mode: str = "relative"  # or "absolute"\n'
            "    control_freq: int = 20  # robosuite OSC control frequency (Hz-mismatch experiments)\n",
        ),
        (
            '        if self.task_ids is not None:\n            kwargs["task_ids"] = self.task_ids\n',
            '        kwargs["control_freq"] = self.control_freq\n'
            '        if self.task_ids is not None:\n            kwargs["task_ids"] = self.task_ids\n',
        ),
    ],
    "envs/libero.py": [
        (
            '        control_mode: str = "relative",\n        is_libero_plus: bool = False,\n',
            '        control_mode: str = "relative",\n        control_freq: int = 20,\n'
            "        is_libero_plus: bool = False,\n",
        ),
        (
            "        self.control_mode = control_mode\n",
            "        self.control_mode = control_mode\n        self.control_freq = control_freq\n",
        ),
        (
            "        env = OffScreenRenderEnv(\n"
            "            bddl_file_name=self._task_bddl_file,\n"
            "            camera_heights=self.observation_height,\n"
            "            camera_widths=self.observation_width,\n"
            "        )\n",
            "        env = OffScreenRenderEnv(\n"
            "            bddl_file_name=self._task_bddl_file,\n"
            "            camera_heights=self.observation_height,\n"
            "            camera_widths=self.observation_width,\n"
            "            control_freq=self.control_freq,\n"
            "        )\n",
        ),
    ],
}


def main(root):
    for rel, edits in EDITS.items():
        path = os.path.join(root, rel)
        src = open(path).read()
        if "control_freq" in src:
            print(f"{rel}: already patched")
            continue
        for anchor, repl in edits:
            assert anchor in src, f"anchor not found in {rel}:\n{anchor!r}"
            src = src.replace(anchor, repl, 1)
        open(path, "w").write(src)
        print(f"{rel}: patched")


if __name__ == "__main__":
    main(sys.argv[1])
