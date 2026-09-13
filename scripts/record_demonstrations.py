#!/usr/bin/env python3
"""Documented entry point for recording through the deterministic ROS bridge.

The bridge must be launched separately with `record_dataset:=true`; this
script intentionally does not auto-generate episodes or start a viewer.
"""

from pathlib import Path


if __name__ == "__main__":
    print("Launch mujoco_bridge with record_dataset:=true, then publish a deterministic command.")
    print("Default output: artifacts/datasets/episode_<episode_id>/")
