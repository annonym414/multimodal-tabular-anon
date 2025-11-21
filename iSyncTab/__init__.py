# __init__.py for the iSyncTab package

# ----------------------------
# Default: simpler iSyncTab with NS-PFS
# ----------------------------
from .iSyncTab import (
    iSyncTab as iSyncTab,   # default model
    NSPFS_GPU,
    set_seed as set_seed,
)

# ----------------------------
# Optional: refined /more theory-aligned iSyncTab
# ----------------------------
from .iSyncTab_refined import (
    iSyncTab as iSyncTabRefined,   # alternate model
    NSPFSRefined,
)

__all__ = [
    # default variant
    "iSyncTab",
    "NSPFS_GPU",
    "set_seed",

    # refined variant
    "iSyncTabRefined",
    "NSPFSRefined",
]