"""Idempotently register smolvla_spline in lerobot's policy factory AND the
policies package __init__ (draccus builds the --policy.type choice list from
configs imported at CLI startup; factory registration alone is not enough).

Run: python patch_factory.py /path/to/lerobot/src/lerobot/policies/factory.py
(the sibling __init__.py is patched automatically)
"""
import os
import sys

CLASS_ANCHOR = '''    elif name == "smolvla":
        from .smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
'''
CLASS_ADD = '''    elif name == "smolvla_spline":
        from .smolvla_spline.modeling_smolvla_spline import SmolVLASplinePolicy

        return SmolVLASplinePolicy
'''
CFG_ANCHOR = '''    elif policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
'''
CFG_ADD = '''    elif policy_type == "smolvla_spline":
        from .smolvla_spline.configuration_smolvla_spline import SmolVLASplineConfig

        return SmolVLASplineConfig(**kwargs)
'''


INIT_ANCHOR = "from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig\n"
INIT_ADD = (
    "from .smolvla_spline.configuration_smolvla_spline import "
    "SmolVLASplineConfig as SmolVLASplineConfig\n"
)


def _patch(path, edits, tag):
    src = open(path).read()
    if "smolvla_spline" in src:
        print(f"{tag} already patched")
        return
    for anchor, add in edits:
        assert anchor in src, f"anchor not found in {tag} — lerobot layout changed"
        src = src.replace(anchor, anchor + add, 1)
    open(path, "w").write(src)
    print(f"{tag} patched")


CLASS_ADD2 = '''    elif name == "smolvla_interp":
        from .smolvla_spline.smolvla_interp import SmolVLAInterpPolicy

        return SmolVLAInterpPolicy
'''
CFG_ADD2 = '''    elif policy_type == "smolvla_interp":
        from .smolvla_spline.smolvla_interp import SmolVLAInterpConfig

        return SmolVLAInterpConfig(**kwargs)
'''
INIT_ADD2 = (
    "from .smolvla_spline.smolvla_interp import "
    "SmolVLAInterpConfig as SmolVLAInterpConfig\n"
)


def _patch2(path, edits, tag, marker):
    src = open(path).read()
    if marker in src:
        print(f"{tag} already has {marker}")
        return
    for anchor, add in edits:
        assert anchor in src, f"anchor not found in {tag}"
        src = src.replace(anchor, anchor + add, 1)
    open(path, "w").write(src)
    print(f"{tag}: {marker} registered")


def main(path):
    init_path = os.path.join(os.path.dirname(path), "__init__.py")
    _patch(path, [(CLASS_ANCHOR, CLASS_ADD), (CFG_ANCHOR, CFG_ADD)], "factory.py")
    _patch(init_path, [(INIT_ANCHOR, INIT_ADD)], "policies/__init__.py")
    _patch2(path, [(CLASS_ANCHOR, CLASS_ADD2), (CFG_ANCHOR, CFG_ADD2)], "factory.py", "smolvla_interp")
    _patch2(init_path, [(INIT_ANCHOR, INIT_ADD2)], "policies/__init__.py", "smolvla_interp")


if __name__ == "__main__":
    main(sys.argv[1])
