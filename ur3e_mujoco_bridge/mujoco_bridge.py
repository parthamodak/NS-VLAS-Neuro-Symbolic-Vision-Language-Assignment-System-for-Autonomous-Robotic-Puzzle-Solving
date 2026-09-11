"""ROS 2 entry point for the existing UR3e MuJoCo pick demo."""

from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .ik import ARM_ACTUATORS, GRIPPER_ACTUATORS, named_actuator_ids, set_gripper
from .task_controller import PickTaskController



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
    """Single-process ROS 2 node; command parsing is a mock language-action interface."""

    def __init__(self) -> None:
        super().__init__("ur3e_mujoco_bridge")
        self.declare_parameter("scene_path", DEFAULT_SCENE)
        scene_path = Path(self.get_parameter("scene_path").value)
        if not scene_path.is_file():
            raise FileNotFoundError(f"MuJoCo scene not found: {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.joint_state_publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.task_state_publisher = self.create_publisher(String, "/task_state", 10)
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
        self._publish_task_state("IDLE")
        self.get_logger().info(f"Loaded MuJoCo scene: {scene_path}")
        self.get_logger().info("Mock language-action interface ready on /robot_command")

    def _require_name(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Missing MuJoCo {object_type.name}: {name}")
        return object_id

    def _publish_task_state(self, state: str) -> None:
        self.task_state_publisher.publish(String(data=state))
        self.get_logger().info(f"[{state}]")

    def _command_callback(self, message: String) -> None:
        command = message.data.strip().lower()
        self.get_logger().info(f"Received command: {command}")
        if "pick" in command and "cube" in command:
            self.task.start_pick_cube()
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
                self.task.update()
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
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
