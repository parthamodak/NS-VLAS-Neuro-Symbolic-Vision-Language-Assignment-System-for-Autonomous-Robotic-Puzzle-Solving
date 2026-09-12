# NS-VLAS

## Neuro-Symbolic Vision-Language Assignment System for Autonomous Robotic Puzzle Solving

## Overview

NS-VLAS is an MSc Artificial Intelligence and Robotics project at the University
of Hertfordshire. It demonstrates a simulated UR3e robotic arm using ROS 2
Jazzy, MuJoCo, and a custom Python ROS 2–MuJoCo bridge. A deterministic task
controller combines pose-aware damped least-squares inverse kinematics with a
two-finger gripper to complete a physical simulated cube pick-and-lift.

The wider research direction includes neuro-symbolic vision-language task
assignment. VLMs, LLMs, VLAs, reinforcement learning, camera perception, and
real-robot deployment are **not implemented** in this final prototype.

## What this repository demonstrates

- ROS 2 Python node development with `rclpy` publishers and subscribers
- MuJoCo simulation of UR3e manipulation and a two-finger gripper
- pose-aware Damped Least-Squares inverse kinematics
- position and orientation control with bounded actuator commands
- deterministic state-machine task execution
- physical contact/friction grasping without an artificial weld
- debugging of gripper geometry, wrist collision proxies, and grasp timing

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
        ↓
Cube manipulation
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

## Quick Start

Terminal 1 — build and run:

```bash
cd ~/ur3e_ws
source /opt/ros/jazzy/setup.bash
export MAKEFLAGS="-j1"
export CMAKE_BUILD_PARALLEL_LEVEL=1
colcon build --symlink-install --packages-select ur3e_mujoco_bridge --executor sequential
source install/setup.bash
ros2 run ur3e_mujoco_bridge mujoco_bridge
```

Terminal 2 — send the predefined command:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ur3e_ws/install/setup.bash
ros2 topic pub --once /robot_command std_msgs/msg/String "{data: 'pick up the cube'}"
```

The `-j1` and sequential settings are especially useful on low-memory systems.

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
