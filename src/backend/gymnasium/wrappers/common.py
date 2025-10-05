"""Minimal wrappers.common module with OrderEnforcing wrapper used in tests."""

from __future__ import annotations


class OrderEnforcing:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name: str):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)
