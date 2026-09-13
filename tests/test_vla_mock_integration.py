"""Headless integration test for the explicit, TEST-ONLY mock VLA path."""

import unittest

import numpy as np
import rclpy
from std_msgs.msg import String

from ur3e_mujoco_bridge.mujoco_bridge import MujocoBridge


class TestVLAMockIntegration(unittest.TestCase):
    def test_mock_observation_to_dls(self):
        rclpy.init(args=["--ros-args", "-p", "policy_mode:=vla", "-p", "vla_backend:=mock"])
        node = MujocoBridge()
        try:
            node._command_callback(String(data="pick up the cube"))
            node._update_vla()
            self.assertEqual(node.policy_mode, "vla")
            self.assertEqual(node.vla_instruction, "pick up the cube")
            self.assertTrue(np.isfinite(node.data.ctrl).all())
        finally:
            if node.vla_camera is not None:
                node.vla_camera.close()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
