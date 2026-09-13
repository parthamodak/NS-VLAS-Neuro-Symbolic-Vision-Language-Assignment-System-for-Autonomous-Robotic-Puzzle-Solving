"""Lazy adapter for the current official LeRobot SmolVLA API.

No ML package is imported until this backend is selected in VLA mode. A custom
UR3e checkpoint must expose compatible state/image/action features; the base
checkpoint is not treated as a deployable UR3e policy.
"""

import time

import numpy as np

from .action import PROJECT_ACTION_DIMENSION, ProjectAction
from .base_policy import BaseVLAPolicy
from .observation import VLAObservation


class SmolVLAPolicy(BaseVLAPolicy):
    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        self.checkpoint = checkpoint
        self.device = device
        self._policy = None
        self._preprocessor = None
        self.last_latency_seconds: float | None = None

    def _load(self) -> None:
        if self._policy is not None:
            return
        try:
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as LeRobotSmolVLA
        except ImportError as error:
            raise RuntimeError(
                "SmolVLA requires a separate Python 3.12 LeRobot environment; "
                "see requirements-vla.txt and training/README.md"
            ) from error
        self._torch = torch
        self._policy = LeRobotSmolVLA.from_pretrained(self.checkpoint).to(self.device)
        self._policy.eval()
        self._preprocessor, _ = make_pre_post_processors(
            self._policy.config, pretrained_path=self.checkpoint
        )

    def predict(self, observation: VLAObservation) -> list[ProjectAction]:
        self._load()
        torch = self._torch
        assert self._policy is not None and self._preprocessor is not None
        # LeRobot uses flat observation keys. The checkpoint must have been
        # trained with this 8-D UR3e state and vla_camera image feature.
        batch = {
            "observation.state": torch.from_numpy(observation.robot_state).unsqueeze(0),
            "observation.images.vla_camera": torch.from_numpy(observation.image)
            .permute(2, 0, 1)
            .unsqueeze(0),
            "task": [observation.language_instruction],
        }
        started = time.perf_counter()
        processed = self._preprocessor(batch)
        with torch.no_grad():
            chunk = self._policy.predict_action_chunk(processed)
        self.last_latency_seconds = time.perf_counter() - started
        values = chunk.detach().float().cpu().numpy()[0]
        if values.ndim != 2 or values.shape[1] != PROJECT_ACTION_DIMENSION:
            raise RuntimeError(
                "SmolVLA checkpoint action dimension is incompatible with NS-VLAS project actions: "
                f"expected {PROJECT_ACTION_DIMENSION}, got {values.shape}"
            )
        return [ProjectAction.from_array(action) for action in values]
