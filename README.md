# NS-VLAS: Neuro-Symbolic Vision-Language Assignment System for Autonomous Robotic Puzzle Solving

## Overview

NS-VLAS is an MSc Artificial Intelligence and Robotics project at the
University of Hertfordshire. This repository currently provides a ROS 2 and
MuJoCo UR3e manipulation foundation: a custom Python bridge, deterministic
task controller, pose-aware damped least-squares (DLS) inverse kinematics, and
a two-finger gripper that performs a simulated physical cube pick-and-lift.

The wider research direction includes neuro-symbolic vision-language task
assignment. VLMs, LLMs, VLAs, reinforcement learning, camera perception, and
real-robot deployment are **not implemented** in this final prototype.

## What is implemented

- ROS 2 Jazzy Python package (`ur3e_mujoco_bridge`)
- MuJoCo UR3e scene with bundled mesh assets
- `/robot_command` `std_msgs/String` command subscriber
- `/joint_states` `sensor_msgs/JointState` publisher
- `/task_state` publisher for deterministic task-state reporting
- pose-aware weighted DLS IK with damping and bounded joint increments
- physical two-finger grasp using MuJoCo contact and friction, without a weld

## Architecture

```text
ROS 2 String command
        ↓
/robot_command
        ↓
mujoco_bridge.py
        ↓
PickTaskController
        ↓
APPROACH → DESCEND → GRASP → LIFT
        ↓
DLS IK
        ↓
MuJoCo actuator commands
        ↓
UR3e + two-finger gripper
```

Command interpretation is predefined keyword matching, not natural-language
AI. For example, a command containing both `pick` and `cube` starts the
pick-and-lift controller.

## Repository structure

```text
.
├── simulation/
│   ├── scene.xml                 # Portable MuJoCo scene
│   ├── assets/                   # UR3e OBJ mesh assets
│   └── LICENSE.txt               # Upstream BSD-3-Clause asset licence
├── ur3e_mujoco_bridge/
│   ├── mujoco_bridge.py          # ROS 2 / MuJoCo integration node
│   ├── task_controller.py        # Deterministic pick state machine
│   └── ik.py                     # Pose-aware weighted DLS IK
├── resource/ur3e_mujoco_bridge
├── package.xml
├── setup.py
├── TEST_RESULTS.md
└── demo/README.md
```

## Requirements

- Ubuntu 24.04 or a compatible Linux environment (WSL2 was used for development)
- ROS 2 Jazzy
- Python 3
- MuJoCo Python bindings (`mujoco`)
- NumPy
- `colcon` / `ament_python`

The ROS package metadata declares `rclpy`, `sensor_msgs`, `std_msgs`, and
`ament_index_python`. MuJoCo and NumPy are imported directly by the Python
source and must be available in the Python environment.

## Installation

Create a ROS 2 workspace and clone this repository as the package directory:

```bash
mkdir -p ~/ur3e_ws/src
git clone https://github.com/parthamodak/NS-VLAS-Neuro-Symbolic-Vision-Language-Assignment-System-for-Autonomous-Robotic-Puzzle-Solving.git \
  ~/ur3e_ws/src/ur3e_mujoco_bridge
```

The bundled `simulation/scene.xml` and `simulation/assets/` are installed with
the package. The default scene is portable; an alternative scene can still be
provided through the `scene_path` ROS parameter.

## Build

```bash
cd ~/ur3e_ws
source /opt/ros/jazzy/setup.bash
export MAKEFLAGS="-j1"
export CMAKE_BUILD_PARALLEL_LEVEL=1
colcon build --symlink-install --packages-select ur3e_mujoco_bridge --executor sequential
source install/setup.bash
```

The `-j1` and sequential settings are especially useful on low-memory systems.

## Run

Terminal 1:

```bash
cd ~/ur3e_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run ur3e_mujoco_bridge mujoco_bridge
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ur3e_ws/install/setup.bash
ros2 topic pub --once /robot_command std_msgs/msg/String "{data: 'pick up the cube'}"
```

To override the packaged scene:

```bash
ros2 run ur3e_mujoco_bridge mujoco_bridge --ros-args -p scene_path:=/absolute/path/to/scene.xml
```

## Supported commands

| Command | Effect |
| --- | --- |
| `pick up the cube` | Starts a cube pick-and-lift |
| `pick cube` | Starts a cube pick-and-lift |
| `open gripper` | Opens the gripper |
| `close gripper` | Closes the gripper |
| `reset` | Resets MuJoCo data and task state |

## State machine

```text
IDLE → APPROACH → DESCEND → GRASP → LIFT → SUCCESS
```

`FAILED` is also reported when an approach, descend, or lift phase exceeds its
configured timeout.

## Results

See [TEST_RESULTS.md](TEST_RESULTS.md) for the recorded physical MuJoCo
pick-and-lift result.

## Limitations

- Simulation only; physical UR3e deployment has not been validated.
- Object pose is read from MuJoCo ground truth rather than camera perception.
- Commands use predefined keyword logic.
- VLM, LLM, VLA, and reinforcement learning are not implemented in this
  prototype.

## Future work

Future work includes camera-based object perception, language grounding,
neuro-symbolic task assignment, VLM/LLM/VLA integration, policy learning, and
safe physical UR3e deployment.

## Demo

See [demo/README.md](demo/README.md). Demo video will be added.

## Licence and third-party assets

The bundled UR3e mesh assets retain the upstream BSD-3-Clause licence in
[`simulation/LICENSE.txt`](simulation/LICENSE.txt).
