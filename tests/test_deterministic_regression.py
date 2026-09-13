"""Headless regression check for the original physical pick-and-lift baseline."""

from pathlib import Path
import unittest

import mujoco

from ur3e_mujoco_bridge.ik import ARM_ACTUATORS, GRIPPER_ACTUATORS, named_actuator_ids, set_gripper
from ur3e_mujoco_bridge.task_controller import PickTaskController, TaskState


class TestDeterministicRegression(unittest.TestCase):
    def test_pick_and_lift_reaches_success(self):
        scene = Path(__file__).resolve().parents[1] / "simulation" / "scene.xml"
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        arm_ids = named_actuator_ids(model, ARM_ACTUATORS)
        gripper_ids = named_actuator_ids(model, GRIPPER_ACTUATORS)
        set_gripper(data, gripper_ids, closing=False)
        controller = PickTaskController(
            model, data,
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cube_target"),
            arm_ids, gripper_ids,
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad"),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad"),
            ],
            lambda state: None, lambda message: None,
        )
        initial_z = float(data.site_xpos[controller.cube_site_id][2])
        controller.start_pick_cube()
        for _ in range(70000):
            controller.update()
            mujoco.mj_step(model, data)
            if controller.state in {TaskState.SUCCESS, TaskState.FAILED}:
                break
        self.assertEqual(controller.state, TaskState.SUCCESS)
        self.assertGreater(float(data.site_xpos[controller.cube_site_id][2]), initial_z + 0.08)


if __name__ == "__main__":
    unittest.main()
