"""Small deterministic pick-and-lift state machine (a mock language-action layer)."""

from enum import Enum

import mujoco
import numpy as np

from .ik import POSITION_TOLERANCE, move_end_effector, set_gripper


class TaskState(str, Enum):
    IDLE = "IDLE"
    APPROACH = "APPROACH"
    DESCEND = "DESCEND"
    GRASP = "GRASP"
    LIFT = "LIFT"
    SUCCESS = "SUCCESS"


class PickTaskController:
    """Runs the existing position-only IK in the original pick-test phases."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        attachment_site_id: int,
        cube_body_id: int,
        cube_site_id: int,
        arm_ids: list[int],
        gripper_ids: list[int],
        on_state_change,
    ) -> None:
        self.model = model
        self.data = data
        self.attachment_site_id = attachment_site_id
        self.cube_body_id = cube_body_id
        self.cube_site_id = cube_site_id
        self.arm_ids = arm_ids
        self.gripper_ids = gripper_ids
        self.on_state_change = on_state_change
        self.state = TaskState.IDLE
        self.phase_start_time = 0.0
        self.initial_cube_z = 0.0
        self.targets: dict[TaskState, np.ndarray] = {}
        self.manual_gripper_closing = False

    def _cube_position(self) -> np.ndarray:
        return (
            self.data.site_xpos[self.cube_site_id].copy()
            if self.cube_site_id >= 0
            else self.data.xpos[self.cube_body_id].copy()
        )

    def _set_state(self, state: TaskState) -> None:
        if self.state != state:
            self.state = state
            self.phase_start_time = self.data.time
            self.on_state_change(state.value)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.manual_gripper_closing = False
        set_gripper(self.data, self.gripper_ids, closing=False)
        self.targets = {}
        self._set_state(TaskState.IDLE)

    def start_pick_cube(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        cube_position = self._cube_position()
        self.initial_cube_z = float(cube_position[2])
        self.targets = {
            TaskState.APPROACH: cube_position + np.array([0.0, 0.0, 0.12]),
            TaskState.DESCEND: cube_position + np.array([0.0, 0.0, 0.015]),
            TaskState.LIFT: cube_position + np.array([0.0, 0.0, 0.20]),
        }
        self.manual_gripper_closing = False
        set_gripper(self.data, self.gripper_ids, closing=False)
        self._set_state(TaskState.APPROACH)

    def open_gripper(self) -> None:
        self.manual_gripper_closing = False
        set_gripper(self.data, self.gripper_ids, closing=False)
        self._set_state(TaskState.IDLE)

    def close_gripper(self) -> None:
        self.manual_gripper_closing = True
        set_gripper(self.data, self.gripper_ids, closing=True)
        self._set_state(TaskState.IDLE)

    def update(self) -> None:
        if self.state == TaskState.IDLE:
            set_gripper(self.data, self.gripper_ids, closing=self.manual_gripper_closing)
            return

        if self.state == TaskState.SUCCESS:
            set_gripper(self.data, self.gripper_ids, closing=True)
            return

        if self.state == TaskState.APPROACH:
            set_gripper(self.data, self.gripper_ids, closing=False)
            if move_end_effector(
                self.model, self.data, self.attachment_site_id, self.arm_ids, self.targets[TaskState.APPROACH]
            ) < POSITION_TOLERANCE:
                self._set_state(TaskState.DESCEND)
            return

        if self.state == TaskState.DESCEND:
            set_gripper(self.data, self.gripper_ids, closing=False)
            if move_end_effector(
                self.model, self.data, self.attachment_site_id, self.arm_ids, self.targets[TaskState.DESCEND]
            ) < POSITION_TOLERANCE:
                self._set_state(TaskState.GRASP)
            return

        if self.state == TaskState.GRASP:
            set_gripper(self.data, self.gripper_ids, closing=True)
            if self.data.time - self.phase_start_time >= 1.0:
                self._set_state(TaskState.LIFT)
            return

        if self.state == TaskState.LIFT:
            set_gripper(self.data, self.gripper_ids, closing=True)
            distance = move_end_effector(
                self.model, self.data, self.attachment_site_id, self.arm_ids, self.targets[TaskState.LIFT]
            )
            cube_lifted = self._cube_position()[2] > self.initial_cube_z + 0.08
            if distance < POSITION_TOLERANCE and cube_lifted:
                self._set_state(TaskState.SUCCESS)
