"""Strict validation and conservative clipping for learned-policy actions."""

from dataclasses import dataclass

import numpy as np

from .action import ProjectAction


@dataclass(frozen=True)
class SafetyLimits:
    max_translation_step: float = 0.01
    max_rotation_step: float = 0.10
    workspace_min: tuple[float, float, float] = (-0.65, -0.55, 0.08)
    workspace_max: tuple[float, float, float] = (0.65, 0.55, 0.65)
    gripper_min: float = 0.0
    gripper_max: float = 0.045


class ActionSafety:
    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits

    def validate_and_clip(self, action: ProjectAction, current_position: np.ndarray) -> ProjectAction:
        values = action.as_array()
        if not np.all(np.isfinite(values)):
            raise ValueError("VLA action contains NaN or infinity")
        position = np.asarray(current_position, dtype=np.float64).reshape(-1)
        if position.size != 3 or not np.all(np.isfinite(position)):
            raise ValueError("Current end-effector position is invalid")
        translation = np.clip(values[:3], -self.limits.max_translation_step, self.limits.max_translation_step)
        rotation = np.clip(values[3:6], -self.limits.max_rotation_step, self.limits.max_rotation_step)
        proposed = position + translation
        bounded = np.clip(
            proposed,
            np.asarray(self.limits.workspace_min),
            np.asarray(self.limits.workspace_max),
        )
        # Clip again after workspace projection to suppress floating-point
        # overshoot at an otherwise exact step boundary.
        translation = np.clip(
            bounded - position,
            -self.limits.max_translation_step,
            self.limits.max_translation_step,
        )
        gripper = float(np.clip(values[6], self.limits.gripper_min, self.limits.gripper_max))
        return ProjectAction.from_array(np.concatenate((translation, rotation, [gripper])))
