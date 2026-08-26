"""Damped least-squares position IK reused from python_main_pick.py."""

import mujoco
import numpy as np


ARM_ACTUATORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
GRIPPER_ACTUATORS = ("left_gripper", "right_gripper")
POSITION_TOLERANCE = 0.008
MAX_JOINT_STEP = 0.035
DAMPING = 0.05
ORIENTATION_WEIGHT = 1.0
# Verified from the MuJoCo pad positions: 0.0 is open, 0.045 is closed.
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.045


def named_actuator_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> list[int]:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]
    missing = [name for name, actuator_id in zip(names, ids) if actuator_id < 0]
    if missing:
        raise ValueError(f"Missing actuator(s): {', '.join(missing)}")
    return ids


def set_gripper(data: mujoco.MjData, gripper_ids: list[int], closing: bool) -> None:
    value = GRIPPER_CLOSED if closing else GRIPPER_OPEN
    for actuator_id in gripper_ids:
        data.ctrl[actuator_id] = value


def move_end_effector(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    attachment_site_id: int,
    arm_actuator_ids: list[int],
    target_position: np.ndarray,
    target_rotation: np.ndarray | None = None,
) -> float:
    """Apply one stable DLS Cartesian-position IK update and return distance."""
    mujoco.mj_forward(model, data)
    error = target_position - data.site_xpos[attachment_site_id]
    distance = float(np.linalg.norm(error))

    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, attachment_site_id)
    arm_dof_ids = np.array(
        [model.jnt_dofadr[model.actuator_trnid[actuator_id, 0]] for actuator_id in arm_actuator_ids]
    )
    jacobian = jacobian_position[:, arm_dof_ids]
    weighted_error = error
    if target_rotation is not None:
        current_quaternion = np.empty(4)
        target_quaternion = np.empty(4)
        rotation_error = np.empty(3)
        mujoco.mju_mat2Quat(current_quaternion, data.site_xmat[attachment_site_id])
        mujoco.mju_mat2Quat(target_quaternion, target_rotation)
        mujoco.mju_subQuat(rotation_error, target_quaternion, current_quaternion)
        jacobian = np.vstack((jacobian, ORIENTATION_WEIGHT * jacobian_rotation[:, arm_dof_ids]))
        weighted_error = np.concatenate((error, ORIENTATION_WEIGHT * rotation_error))
    update = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + (DAMPING**2) * np.eye(jacobian.shape[0]), weighted_error
    )
    update = np.clip(update, -MAX_JOINT_STEP, MAX_JOINT_STEP)

    for actuator_id, joint_update in zip(arm_actuator_ids, update):
        joint_id = model.actuator_trnid[actuator_id, 0]
        qpos_index = model.jnt_qposadr[joint_id]
        ctrl_min, ctrl_max = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(
            data.qpos[qpos_index] + joint_update, ctrl_min, ctrl_max
        )
    return distance
