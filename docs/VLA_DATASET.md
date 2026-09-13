# Native demonstration dataset

NS-VLAS initially uses a simple native format instead of coupling the stable
ROS runtime to a rapidly evolving LeRobot dataset API. Each `episode_<id>` has
PNG RGB frames and `metadata.json`: complete instruction, eight-value state,
eight actuator control targets, timestep, timestamp, and final success. Cube
coordinates are evaluation metadata only, never visual-policy input.

Validate it with `python3 scripts/validate_dataset.py artifacts/datasets`.
A converter to LeRobot format is intentionally deferred until a compatible
Python 3.12 LeRobot environment and final action schema are selected.
