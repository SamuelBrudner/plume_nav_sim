"""Video plume adapter placeholder.

The real VideoPlumeAdapter requires optional video-processing dependencies and
sample plume videos. These are not available in the test environment, so the
adapter is disabled. Tests relying on this component will fall back to a
lightweight stub.
"""

raise ImportError("VideoPlumeAdapter optional dependencies not installed")
