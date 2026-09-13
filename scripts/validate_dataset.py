#!/usr/bin/env python3
"""Validate the lightweight native NS-VLAS episode format."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    episodes = sorted(args.dataset.glob("episode_*/metadata.json"))
    if not episodes:
        raise SystemExit("No episode_*/metadata.json files found")
    frame_count = 0
    for metadata_path in episodes:
        metadata = json.loads(metadata_path.read_text())
        assert metadata.get("instruction"), f"Missing instruction: {metadata_path}"
        assert "success" in metadata, f"Missing success: {metadata_path}"
        records = metadata.get("records", [])
        assert records, f"No records: {metadata_path}"
        for record in records:
            image_path = metadata_path.parent / record["image"]
            image = np.asarray(Image.open(image_path).convert("RGB"))
            state = np.asarray(record["robot_state"], dtype=float)
            action = np.asarray(record["actuator_control"], dtype=float)
            assert image.ndim == 3 and image.shape[2] == 3
            assert state.shape == (8,) and action.shape == (8,)
            assert np.isfinite(state).all() and np.isfinite(action).all()
            frame_count += 1
    print(f"valid episodes={len(episodes)} frames={frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
