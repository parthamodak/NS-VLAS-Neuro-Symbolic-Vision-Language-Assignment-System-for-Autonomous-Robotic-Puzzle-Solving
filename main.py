import mujoco
import mujoco.viewer
import time

# Load your MuJoCo model
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Open MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # Open gripper
        data.ctrl[6] = 0.045
        data.ctrl[7] = 0.045

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(0.002)