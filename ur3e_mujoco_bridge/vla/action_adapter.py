"""Converts safe Cartesian actions into targets for the existing DLS IK."""

from dataclasses import dataclass

import numpy as np

from .action import ProjectAction


@dataclass(frozen=True)
class AdaptedAction:
    target_position: np.ndarray
    target_rotation: np.ndarray
    gripper: float


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def adapt_action(
    action: ProjectAction, current_position: np.ndarray, current_rotation: np.ndarray
) -> AdaptedAction:
    """Apply local roll/pitch/yaw deltas to the current attachment-site pose."""
    current_position = np.asarray(current_position, dtype=np.float64).reshape(3)
    current_rotation = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
    local_delta = _rotation_x(action.delta_roll) @ _rotation_y(action.delta_pitch) @ _rotation_z(
        action.delta_yaw
    )
    return AdaptedAction(
        target_position=current_position + action.as_array()[:3],
        # MuJoCo site/orientation APIs use a flat, row-major 3x3 matrix.
        target_rotation=(current_rotation @ local_delta).reshape(9),
        gripper=action.gripper,
    )
