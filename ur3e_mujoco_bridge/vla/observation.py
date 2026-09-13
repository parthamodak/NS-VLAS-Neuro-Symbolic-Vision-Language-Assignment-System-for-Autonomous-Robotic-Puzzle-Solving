"""MuJoCo-backed robot observation construction, without object ground truth."""

from dataclasses import dataclass

import mujoco
import numpy as np


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_JOINT_NAMES = ("left_finger_joint", "right_finger_joint")
ROBOT_STATE_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES


@dataclass(frozen=True)
class VLAObservation:
    """One policy observation. RGB is the only visual object input."""

    image: np.ndarray
    robot_state: np.ndarray
    language_instruction: str


def robot_state_from_mujoco(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return qpos in the documented, model-name-derived eight-value order."""
    values: list[float] = []
    for joint_name in ROBOT_STATE_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Missing robot joint required for VLA state: {joint_name}")
        values.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
    return np.asarray(values, dtype=np.float32)


def build_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    image: np.ndarray,
    language_instruction: str,
) -> VLAObservation:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("VLA image must be an HxWx3 uint8 RGB array")
    instruction = language_instruction.strip()
    if not instruction:
        raise ValueError("VLA mode requires a non-empty language instruction")
    return VLAObservation(image, robot_state_from_mujoco(model, data), instruction)
