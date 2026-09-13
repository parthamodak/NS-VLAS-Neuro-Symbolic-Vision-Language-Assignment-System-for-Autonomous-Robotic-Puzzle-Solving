"""Project-side Cartesian action representation.

This is deliberately distinct from a model's native action schema.  The first
six values are local end-effector deltas (metres, radians); the final value is
the MuJoCo gripper position in metres.
"""

from dataclasses import dataclass

import numpy as np


PROJECT_ACTION_DIMENSION = 7


@dataclass(frozen=True)
class ProjectAction:
    delta_x: float
    delta_y: float
    delta_z: float
    delta_roll: float
    delta_pitch: float
    delta_yaw: float
    gripper: float

    @classmethod
    def from_array(cls, values: np.ndarray | list[float]) -> "ProjectAction":
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != PROJECT_ACTION_DIMENSION:
            raise ValueError(
                f"Expected {PROJECT_ACTION_DIMENSION} project action values, got {array.size}"
            )
        return cls(*(float(value) for value in array))

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.delta_x,
                self.delta_y,
                self.delta_z,
                self.delta_roll,
                self.delta_pitch,
                self.delta_yaw,
                self.gripper,
            ],
            dtype=np.float64,
        )
