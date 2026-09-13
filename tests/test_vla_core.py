import unittest

import numpy as np

from ur3e_mujoco_bridge.vla.action import ProjectAction
from ur3e_mujoco_bridge.vla.action_adapter import adapt_action
from ur3e_mujoco_bridge.vla.mock_policy import MockPolicy
from ur3e_mujoco_bridge.vla.observation import VLAObservation
from ur3e_mujoco_bridge.vla.safety import ActionSafety, SafetyLimits


class TestVLACore(unittest.TestCase):
    def test_safety_rejects_nan(self):
        safety = ActionSafety(SafetyLimits())
        with self.assertRaises(ValueError):
            safety.validate_and_clip(ProjectAction(np.nan, 0, 0, 0, 0, 0, 0), np.zeros(3))

    def test_safety_clips_workspace_and_steps(self):
        action = ActionSafety(SafetyLimits()).validate_and_clip(
            ProjectAction(10, -10, 10, 10, -10, 10, 1), np.array([0.64, -0.54, 0.64])
        )
        self.assertTrue(np.all(np.abs(action.as_array()[:6]) <= np.array([.01, .01, .01, .1, .1, .1])))
        self.assertEqual(action.gripper, .045)

    def test_adapter_and_mock(self):
        target = adapt_action(ProjectAction(.01, 0, 0, 0, 0, 0, 0), np.zeros(3), np.eye(3))
        np.testing.assert_allclose(target.target_position, [.01, 0, 0])
        observation = VLAObservation(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros(8, dtype=np.float32), "pick up the cube")
        self.assertEqual(len(MockPolicy().predict(observation)), 1)


if __name__ == "__main__":
    unittest.main()
