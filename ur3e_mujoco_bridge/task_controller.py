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
    FAILED = "FAILED"


PHASE_TIMEOUTS = {
    TaskState.APPROACH: 35.0,
    TaskState.DESCEND: 25.0,
    TaskState.LIFT: 30.0,
}
DEBUG_PERIOD = 1.0
APPROACH_TOLERANCE = 0.015
PREGRASP_SETTLE_TIME = 1.0


class PickTaskController:
    """Deterministic physical pick-and-lift state machine using pose-aware IK."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        attachment_site_id: int,
        cube_body_id: int,
        cube_site_id: int,
        arm_ids: list[int],
        gripper_ids: list[int],
        grasp_geom_ids: list[int],
        on_state_change,
        on_debug,
    ) -> None:
        self.model = model
        self.data = data
        self.attachment_site_id = attachment_site_id
        self.cube_body_id = cube_body_id
        self.cube_site_id = cube_site_id
        self.arm_ids = arm_ids
        self.gripper_ids = gripper_ids
        self.grasp_geom_ids = grasp_geom_ids
        self.on_state_change = on_state_change
        self.on_debug = on_debug
        self.state = TaskState.IDLE
        self.phase_start_time = 0.0
        self.initial_cube_z = 0.0
        self.targets: dict[TaskState, np.ndarray] = {}
        self.manual_gripper_closing = False
        self.approach_stage = 0
        self.pregrasp_settle_start: float | None = None
        self.target_rotation: np.ndarray | None = None
        self.last_debug_time = -DEBUG_PERIOD

    def _cube_position(self) -> np.ndarray:
        return (
            self.data.site_xpos[self.cube_site_id].copy()
            if self.cube_site_id >= 0
            else self.data.xpos[self.cube_body_id].copy()
        )

    def _grasp_center(self) -> np.ndarray:
        return np.mean(self.data.geom_xpos[self.grasp_geom_ids], axis=0)

    def _move_grasp_center(
        self, target_position: np.ndarray, target_rotation: np.ndarray | None = None
    ) -> float:
        """Use attachment_site for IK while targeting the centre of the finger pads."""
        mujoco.mj_forward(self.model, self.data)
        attachment_position = self.data.site_xpos[self.attachment_site_id].copy()
        pad_offset = self._grasp_center() - attachment_position
        move_end_effector(
            self.model,
            self.data,
            self.attachment_site_id,
            self.arm_ids,
            target_position - pad_offset,
            target_rotation,
        )
        return float(np.linalg.norm(target_position - self._grasp_center()))

    def _orientation_error(self, target_rotation: np.ndarray | None) -> float | None:
        if target_rotation is None:
            return None
        current_quaternion = np.empty(4)
        target_quaternion = np.empty(4)
        rotation_error = np.empty(3)
        mujoco.mju_mat2Quat(current_quaternion, self.data.site_xmat[self.attachment_site_id])
        mujoco.mju_mat2Quat(target_quaternion, target_rotation)
        mujoco.mju_subQuat(rotation_error, target_quaternion, current_quaternion)
        return float(np.linalg.norm(rotation_error))

    def _debug(
        self,
        target_position: np.ndarray,
        error: float,
        target_rotation: np.ndarray | None = None,
    ) -> None:
        if self.data.time - self.last_debug_time < DEBUG_PERIOD:
            return
        self.last_debug_time = self.data.time
        orientation_error = self._orientation_error(target_rotation)
        orientation_text = (
            "" if orientation_error is None else f" orientation_error={orientation_error:.4f}"
        )
        self.on_debug(
            f"state={self.state.value} cube={np.round(self._cube_position(), 3)} "
            f"grasp_center={np.round(self._grasp_center(), 3)} "
            f"target={np.round(target_position, 3)} error={error:.4f}{orientation_text}"
        )

    def _timed_out(self) -> bool:
        return self.data.time - self.phase_start_time > PHASE_TIMEOUTS.get(self.state, float("inf"))

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
        self.approach_stage = 0
        self.pregrasp_settle_start = None
        self.target_rotation = None
        self.last_debug_time = -DEBUG_PERIOD
        self._set_state(TaskState.IDLE)

    def start_pick_cube(self) -> None:
        """Create measured approach, grasp-centre, and lift targets for the cube."""
        mujoco.mj_forward(self.model, self.data)
        cube_position = self._cube_position()
        self.initial_cube_z = float(cube_position[2])
        retract_position = self._grasp_center() + np.array([0.0, 0.0, 0.18])
        self.targets = {
            TaskState.APPROACH: retract_position,
            TaskState.DESCEND: cube_position + np.array([0.0, 0.0, 0.025]),
            TaskState.LIFT: cube_position + np.array([0.0, 0.0, 0.20]),
        }
        self.pregrasp_target = cube_position + np.array([0.0, 0.0, 0.25])
        self.manual_gripper_closing = False
        self.approach_stage = 0
        self.pregrasp_settle_start = None
        self.target_rotation = None
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

        if self.state == TaskState.FAILED:
            set_gripper(self.data, self.gripper_ids, closing=False)
            return

        if self.state == TaskState.APPROACH:
            set_gripper(self.data, self.gripper_ids, closing=False)
            error = self._move_grasp_center(self.targets[TaskState.APPROACH])
            self._debug(self.targets[TaskState.APPROACH], error)
            required_tolerance = (
                APPROACH_TOLERANCE if self.approach_stage == 0 else POSITION_TOLERANCE
            )
            if error < required_tolerance:
                if self.approach_stage == 0:
                    self.approach_stage = 1
                    self.targets[TaskState.APPROACH] = self.pregrasp_target
                    self.on_debug("state=APPROACH reached safe retract; moving to pre-grasp")
                elif self.approach_stage == 1:
                    # Let position actuators settle before freezing the grasp orientation.
                    self.approach_stage = 2
                    self.pregrasp_settle_start = self.data.time
                    self.on_debug("state=APPROACH reached pre-grasp; settling before descent")
                elif self.data.time - self.pregrasp_settle_start >= PREGRASP_SETTLE_TIME:
                    self.target_rotation = self.data.site_xmat[self.attachment_site_id].copy()
                    self._set_state(TaskState.DESCEND)
            elif self._timed_out():
                self._set_state(TaskState.FAILED)
            return

        if self.state == TaskState.DESCEND:
            set_gripper(self.data, self.gripper_ids, closing=False)
            error = self._move_grasp_center(
                self.targets[TaskState.DESCEND], self.target_rotation
            )
            self._debug(self.targets[TaskState.DESCEND], error, self.target_rotation)
            if error < POSITION_TOLERANCE:
                self._set_state(TaskState.GRASP)
            elif self._timed_out():
                self._set_state(TaskState.FAILED)
            return

        if self.state == TaskState.GRASP:
            # Hold closure briefly so MuJoCo contact/friction can establish the grasp.
            set_gripper(self.data, self.gripper_ids, closing=True)
            if self.data.time - self.phase_start_time >= 1.0:
                self._set_state(TaskState.LIFT)
            return

        if self.state == TaskState.LIFT:
            set_gripper(self.data, self.gripper_ids, closing=True)
            distance = self._move_grasp_center(self.targets[TaskState.LIFT], self.target_rotation)
            self._debug(self.targets[TaskState.LIFT], distance, self.target_rotation)
            cube_lifted = self._cube_position()[2] > self.initial_cube_z + 0.08
            if distance < POSITION_TOLERANCE and cube_lifted:
                self._set_state(TaskState.SUCCESS)
            elif self._timed_out():
                self._set_state(TaskState.FAILED)
