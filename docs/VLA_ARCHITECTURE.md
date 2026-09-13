# VLA architecture

Deterministic baseline: ROS command → keyword matching → `PickTaskController`
→ target pose → existing DLS IK → MuJoCo.

VLA mode: RGB `vla_camera` frame + complete ROS instruction + eight-value
robot state → selected policy → action chunk queue → safety checks/clipping →
target pose → existing DLS IK → MuJoCo. VLA mode never calls the deterministic
state machine as a hidden fallback. If dependencies/rendering/inference fail,
it reports `/vla/status` `ERROR` and applies no VLA action.

The state order is: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2,
wrist_3, left_finger, right_finger. Cube ground truth is excluded from visual
observation and policy input; it remains valid evaluation metadata.
