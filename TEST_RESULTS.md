# Recorded Test Results

## Deterministic baseline: end-to-end pick-and-lift

Command:

```bash
ros2 topic pub --once /robot_command std_msgs/msg/String "{data: 'pick up the cube'}"
```

Observed state sequence:

```text
IDLE → APPROACH → DESCEND → GRASP → LIFT → SUCCESS
```

The successful recorded ROS 2/MuJoCo execution used MuJoCo physical contact and
friction between the two-finger gripper and cube. No artificial weld or object
attachment constraint was used.

| Measurement | Recorded value |
| --- | --- |
| Initial cube Z | approximately 0.024 m |
| Intermediate cube Z | approximately 0.056 m |
| Intermediate cube Z | approximately 0.094 m |
| Intermediate cube Z | approximately 0.133 m |
| Final cube Z | approximately 0.170 m |
| Vertical displacement | approximately 0.146 m (14.6 cm) |
| Final position error | approximately 0.0127 |
| Final orientation error | approximately 0.0341 |
| Final state | `SUCCESS` |

These values document one successful execution, not a success rate or a
repeated-trial statistical evaluation.

## Component-level checks

The project contains support for and development checks of:

- individual UR3e joint actuation through MuJoCo actuators;
- manual two-finger gripper opening and closing;
- deterministic automatic approach and descent; and
- the full physical pick-and-lift state sequence above.

The standalone `python_main_pick.py` in the original development project was
used as an earlier direct MuJoCo pick-test reference. The supported public
runtime is the ROS 2 bridge documented in the main README.

## VLA infrastructure

- Fixed `vla_camera` rendered a real 224×224 `uint8` RGB frame with pixel
  range 0–255 using MuJoCo `Renderer` under WSL.
- RGB + complete language instruction + eight-value robot state → mock policy
  → safety → existing DLS IK integration was tested headlessly.
- Unit/regression suite: 5 tests passed, including the deterministic physical
  pick-and-lift regression.

## SmolVLA inference and autonomous evaluation

Not yet tested. No LeRobot/SmolVLA dependency, checkpoint, model training, or
real neural inference was installed or run on the development machine.
