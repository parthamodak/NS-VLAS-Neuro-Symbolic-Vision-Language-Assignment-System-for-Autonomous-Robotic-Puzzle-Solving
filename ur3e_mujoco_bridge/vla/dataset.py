"""Small native, inspectable NS-VLAS demonstration recorder (Option B)."""

import json
from pathlib import Path

import numpy as np

from .observation import VLAObservation


class NativeEpisodeRecorder:
    """Writes PNG frames plus aligned JSON metadata; no ML packages required."""

    def __init__(self, output_directory: str | Path, episode_id: str, instruction: str) -> None:
        self.directory = Path(output_directory) / f"episode_{episode_id}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.instruction = instruction
        self.records: list[dict] = []

    def append(
        self, observation: VLAObservation, actuator_control: np.ndarray, timestep: int, timestamp: float
    ) -> None:
        from PIL import Image

        image_name = f"frame_{timestep:06d}.png"
        Image.fromarray(observation.image).save(self.directory / image_name)
        self.records.append(
            {
                "timestep": timestep,
                "timestamp": timestamp,
                "image": image_name,
                "robot_state": observation.robot_state.tolist(),
                "actuator_control": np.asarray(actuator_control, dtype=float).reshape(8).tolist(),
            }
        )

    def finish(self, success: bool, cube_position_evaluation: np.ndarray | None = None) -> Path:
        metadata = {
            "format": "ns_vlas_native_v1",
            "instruction": self.instruction,
            "robot_state_order": [
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                "wrist_2_joint", "wrist_3_joint", "left_finger_joint", "right_finger_joint",
            ],
            "success": bool(success),
            "cube_position_evaluation": None if cube_position_evaluation is None else cube_position_evaluation.tolist(),
            "records": self.records,
        }
        path = self.directory / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2))
        return path
