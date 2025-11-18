"""Vendor namespace for lightweight, vendored third-party shims.

All copied or minimal shim implementations of third-party packages should
live under this namespace (e.g., `vendor.psutil`, `vendor.gymnasium_vendored`).

First-party code should prefer real third-party dependencies. Shims exist
primarily to support tests and constrained environments.
"""
