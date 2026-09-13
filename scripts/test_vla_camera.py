#!/usr/bin/env python3
"""Render and save one RGB frame from the bundled fixed VLA camera."""

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ur3e_mujoco_bridge.vla.camera import MujocoRGBCamera  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=ROOT / "simulation" / "scene.xml")
    parser.add_argument("--camera", default="vla_camera")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "vla_camera_test.png")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = MujocoRGBCamera(model, args.camera, args.width, args.height)
    try:
        frame = camera.render(data)
    finally:
        camera.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required only to save the PNG test frame") from error
    Image.fromarray(frame).save(args.output)
    print(f"camera={args.camera}")
    print(f"width={frame.shape[1]}")
    print(f"height={frame.shape[0]}")
    print(f"dtype={frame.dtype}")
    print(f"min={np.min(frame)} max={np.max(frame)}")
    print(f"saved={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
