"""Reusable MuJoCo RGB camera component."""

import mujoco
import numpy as np


class MujocoRGBCamera:
    """Renders a named fixed MuJoCo camera with the official Renderer API."""

    def __init__(self, model: mujoco.MjModel, camera_name: str, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Camera width and height must be positive")
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"MuJoCo camera not found: {camera_name}")
        self.camera_name = camera_name
        self.width = int(width)
        self.height = int(height)
        self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)

    def render(self, data: mujoco.MjData) -> np.ndarray:
        try:
            self._renderer.update_scene(data, camera=self.camera_name)
            image = np.asarray(self._renderer.render())
        except Exception as error:  # Renderer failures are environment-dependent in WSL.
            raise RuntimeError(f"Could not render MuJoCo camera '{self.camera_name}': {error}") from error
        if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
            raise RuntimeError(
                f"Unexpected RGB frame: shape={image.shape}, dtype={image.dtype}; "
                f"expected ({self.height}, {self.width}, 3) uint8"
            )
        return image

    def close(self) -> None:
        self._renderer.close()
