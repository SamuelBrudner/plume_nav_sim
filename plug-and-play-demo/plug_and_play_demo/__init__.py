"""Demo package containing external (user-land) policies.

This mimics a separate project that depends on plume_nav_sim but defines its
own policy implementations. The run–tumble temporal-derivative policy lives
here to demonstrate plug-and-play composition via dotted-path.
"""

from .stateless_policy import DeltaBasedRunTumblePolicy

__all__ = [
    "DeltaBasedRunTumblePolicy",
]
