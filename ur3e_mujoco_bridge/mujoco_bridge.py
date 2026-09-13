"""ROS 2 entry point for the existing UR3e MuJoCo pick demo."""

from pathlib import Path
import time
from collections import deque

import numpy as np

from ament_index_python.packages import get_package_share_directory
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .ik import ARM_ACTUATORS, GRIPPER_ACTUATORS, named_actuator_ids, set_gripper
from .task_controller import PickTaskController
from .vla.action import ProjectAction
from .vla.action_adapter import adapt_action
from .vla.camera import MujocoRGBCamera
from .vla.dataset import NativeEpisodeRecorder
from .vla.mock_policy import MockPolicy
from .vla.observation import build_observation
from .vla.safety import ActionSafety, SafetyLimits
from .vla.smolvla_policy import SmolVLAPolicy



def _default_scene_path() -> str:
    """Return the bundled scene from source or the installed package share."""
    source_scene = Path(__file__).resolve().parents[1] / "simulation" / "scene.xml"
    if source_scene.is_file():
        return str(source_scene)
    return str(Path(get_package_share_directory("ur3e_mujoco_bridge")) / "simulation" / "scene.xml")


DEFAULT_SCENE = _default_scene_path()
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "left_finger_joint",
    "right_finger_joint",
)
CONTROL_DT = 0.002
PUBLISH_PERIOD = 1.0 / 30.0


class MujocoBridge(Node):
    """Single-process bridge with deterministic and explicitly selected VLA modes."""

    def __init__(self) -> None:
        super().__init__("ur3e_mujoco_bridge")
        self.declare_parameter("scene_path", DEFAULT_SCENE)
        self.declare_parameter("policy_mode", "deterministic")
        self.declare_parameter("vla_backend", "mock")
        self.declare_parameter("vla_checkpoint", "lerobot/smolvla_base")
        self.declare_parameter("vla_device", "cpu")
        self.declare_parameter("vla_camera", "vla_camera")
        self.declare_parameter("vla_image_width", 224)
        self.declare_parameter("vla_image_height", 224)
        self.declare_parameter("vla_frequency_hz", 5.0)
        self.declare_parameter("vla_max_translation_step", 0.01)
        self.declare_parameter("vla_max_rotation_step", 0.10)
        self.declare_parameter("record_dataset", False)
        self.declare_parameter("dataset_directory", "artifacts/datasets")
        self.declare_parameter("episode_id", "001")
        self.declare_parameter("dataset_frequency_hz", 5.0)
        scene_path = Path(self.get_parameter("scene_path").value)
        if not scene_path.is_file():
            raise FileNotFoundError(f"MuJoCo scene not found: {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.joint_state_publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.task_state_publisher = self.create_publisher(String, "/task_state", 10)
        self.vla_status_publisher = self.create_publisher(String, "/vla/status", 10)
        self.vla_action_publisher = self.create_publisher(String, "/vla/action", 10)
        self.command_subscription = self.create_subscription(
            String, "/robot_command", self._command_callback, 10
        )

        self.attachment_site_id = self._require_name(mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.cube_body_id = self._require_name(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cube_target")
        self.joint_ids = [self._require_name(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        grasp_geom_ids = [
            self._require_name(mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad"),
            self._require_name(mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad"),
        ]
        arm_ids = named_actuator_ids(self.model, ARM_ACTUATORS)
        self.gripper_ids = named_actuator_ids(self.model, GRIPPER_ACTUATORS)
        set_gripper(self.data, self.gripper_ids, closing=False)
        mujoco.mj_forward(self.model, self.data)
        self.task = PickTaskController(
            self.model,
            self.data,
            self.attachment_site_id,
            self.cube_body_id,
            self.cube_site_id,
            arm_ids,
            self.gripper_ids,
            grasp_geom_ids,
            self._publish_task_state,
            lambda message: self.get_logger().info(f"[debug] {message}"),
        )
        self.last_joint_publish_time = -PUBLISH_PERIOD
        self.policy_mode = str(self.get_parameter("policy_mode").value).lower()
        if self.policy_mode not in {"deterministic", "vla"}:
            raise ValueError("policy_mode must be 'deterministic' or 'vla'")
        self.vla_frequency_hz = float(self.get_parameter("vla_frequency_hz").value)
        if self.vla_frequency_hz <= 0.0:
            raise ValueError("vla_frequency_hz must be positive")
        limits = SafetyLimits(
            max_translation_step=float(self.get_parameter("vla_max_translation_step").value),
            max_rotation_step=float(self.get_parameter("vla_max_rotation_step").value),
        )
        self.vla_safety = ActionSafety(limits)
        self.vla_camera: MujocoRGBCamera | None = None
        self.vla_policy = None
        self.vla_instruction = ""
        self.vla_actions: deque[ProjectAction] = deque()
        self.last_vla_step_time = -1.0 / self.vla_frequency_hz
        self.recorder: NativeEpisodeRecorder | None = None
        self.last_record_time = -1.0
        self.recording_finished = False
        self._publish_task_state("IDLE")
        self._publish_vla_status("DISABLED" if self.policy_mode == "deterministic" else "READY")
        self.get_logger().info(f"Loaded MuJoCo scene: {scene_path}")
        self.get_logger().info(f"policy_mode={self.policy_mode}; /robot_command ready")

    def _require_name(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Missing MuJoCo {object_type.name}: {name}")
        return object_id

    def _publish_task_state(self, state: str) -> None:
        self.task_state_publisher.publish(String(data=state))
        self.get_logger().info(f"[{state}]")

    def _publish_vla_status(self, status: str) -> None:
        self.vla_status_publisher.publish(String(data=status))
        self.get_logger().info(f"[VLA {status}]")

    def _command_callback(self, message: String) -> None:
        instruction = message.data.strip()
        command = instruction.lower()
        self.get_logger().info(f"Received command: {instruction}")
        if self.policy_mode == "vla":
            if command == "reset":
                self.task.reset()
                self.vla_actions.clear()
                self.vla_instruction = ""
                self._publish_vla_status("READY")
                return
            if not instruction:
                self._publish_vla_status("ERROR: empty language instruction")
                return
            # Preserve the complete user text: VLA mode does not keyword-match it.
            self.vla_instruction = instruction
            self.vla_actions.clear()
            self._publish_vla_status("INITIALIZING")
            return
        if "pick" in command and "cube" in command:
            self.task.start_pick_cube()
            if bool(self.get_parameter("record_dataset").value):
                self._start_recording(instruction)
        elif command == "open gripper":
            self.task.open_gripper()
            self._publish_task_state("GRIPPER_OPEN")
        elif command == "close gripper":
            self.task.close_gripper()
            self._publish_task_state("GRIPPER_CLOSED")
        elif command == "reset":
            self.task.reset()
            self.last_joint_publish_time = -PUBLISH_PERIOD
            self._publish_task_state("RESET")
        else:
            self.get_logger().warning(
                "Supported commands: pick up the cube, pick cube, open gripper, close gripper, reset"
            )

    def _ensure_vla_components(self) -> None:
        if self.vla_camera is None:
            self.vla_camera = MujocoRGBCamera(
                self.model,
                str(self.get_parameter("vla_camera").value),
                int(self.get_parameter("vla_image_width").value),
                int(self.get_parameter("vla_image_height").value),
            )
        if self.vla_policy is None:
            backend = str(self.get_parameter("vla_backend").value).lower()
            if backend == "mock":
                self.vla_policy = MockPolicy()
            elif backend == "smolvla":
                self.vla_policy = SmolVLAPolicy(
                    str(self.get_parameter("vla_checkpoint").value),
                    str(self.get_parameter("vla_device").value),
                )
            else:
                raise ValueError("vla_backend must be 'mock' or 'smolvla'")

    def _update_vla(self) -> None:
        """Run one safe, low-rate VLA action; physics remains at 500 Hz."""
        if not self.vla_instruction:
            return
        if self.data.time - self.last_vla_step_time < 1.0 / self.vla_frequency_hz:
            return
        self.last_vla_step_time = self.data.time
        try:
            self._ensure_vla_components()
            assert self.vla_camera is not None and self.vla_policy is not None
            if not self.vla_actions:
                self._publish_vla_status("LOADING" if isinstance(self.vla_policy, SmolVLAPolicy) else "RUNNING")
                observation = build_observation(
                    self.model,
                    self.data,
                    self.vla_camera.render(self.data),
                    self.vla_instruction,
                )
                actions = self.vla_policy.predict(observation)
                if not actions:
                    raise RuntimeError("VLA policy returned an empty action chunk")
                self.vla_actions.extend(actions)
            raw_action = self.vla_actions.popleft()
            position = self.data.site_xpos[self.attachment_site_id].copy()
            rotation = self.data.site_xmat[self.attachment_site_id].reshape(3, 3).copy()
            action = self.vla_safety.validate_and_clip(raw_action, position)
            target = adapt_action(action, position, rotation)
            from .ik import move_end_effector

            move_end_effector(
                self.model, self.data, self.attachment_site_id, self.task.arm_ids,
                target.target_position, target.target_rotation,
            )
            for actuator_id in self.gripper_ids:
                self.data.ctrl[actuator_id] = target.gripper
            self.vla_action_publisher.publish(String(data=np.array2string(action.as_array(), precision=4)))
            self._publish_vla_status("RUNNING")
        except Exception as error:
            self.vla_actions.clear()
            self.vla_instruction = ""
            self._publish_vla_status(f"ERROR: {error}")

    def _start_recording(self, instruction: str) -> None:
        self.recorder = NativeEpisodeRecorder(
            str(self.get_parameter("dataset_directory").value),
            str(self.get_parameter("episode_id").value),
            instruction,
        )
        self.last_record_time = -1.0
        self.recording_finished = False
        self.get_logger().info("Native NS-VLAS demonstration recording enabled")

    def _record_deterministic_step(self) -> None:
        if self.recorder is None or self.recording_finished:
            return
        frequency = float(self.get_parameter("dataset_frequency_hz").value)
        if frequency <= 0.0 or self.data.time - self.last_record_time < 1.0 / frequency:
            return
        try:
            if self.vla_camera is None:
                self.vla_camera = MujocoRGBCamera(
                    self.model, str(self.get_parameter("vla_camera").value),
                    int(self.get_parameter("vla_image_width").value), int(self.get_parameter("vla_image_height").value),
                )
            observation = build_observation(self.model, self.data, self.vla_camera.render(self.data), self.recorder.instruction)
            self.recorder.append(observation, self.data.ctrl.copy(), len(self.recorder.records), self.data.time)
            self.last_record_time = self.data.time
            if self.task.state.value in {"SUCCESS", "FAILED"}:
                path = self.recorder.finish(
                    self.task.state.value == "SUCCESS", self.data.xpos[self.cube_body_id].copy()
                )
                self.recording_finished = True
                self.get_logger().info(f"Saved demonstration metadata: {path}")
        except Exception as error:
            self.recording_finished = True
            self.get_logger().error(f"Stopped demonstration recording: {error}")

    def _publish_joint_state(self) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = [
            float(self.data.qpos[self.model.jnt_qposadr[joint_id]]) for joint_id in self.joint_ids
        ]
        message.velocity = [
            float(self.data.qvel[self.model.jnt_dofadr[joint_id]]) for joint_id in self.joint_ids
        ]
        self.joint_state_publisher.publish(message)

    def run(self) -> None:
        """Advance ROS callbacks, task control, and MuJoCo in one simulation loop."""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while rclpy.ok() and viewer.is_running():
                # Process queued ROS commands before applying the next control update.
                rclpy.spin_once(self, timeout_sec=0.0)
                if self.policy_mode == "deterministic":
                    self.task.update()
                    self._record_deterministic_step()
                else:
                    self._update_vla()
                mujoco.mj_step(self.model, self.data)
                if self.data.time - self.last_joint_publish_time >= PUBLISH_PERIOD:
                    self._publish_joint_state()
                    self.last_joint_publish_time = self.data.time
                viewer.sync()
                time.sleep(CONTROL_DT)


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = MujocoBridge()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if node.vla_camera is not None:
                node.vla_camera.close()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
