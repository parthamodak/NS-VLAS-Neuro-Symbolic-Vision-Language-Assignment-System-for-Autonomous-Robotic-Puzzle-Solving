import time
import mujoco
import mujoco.viewer


# --------------------------------------------------
# Load the MuJoCo model
# --------------------------------------------------

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)


# --------------------------------------------------
# Gripper settings
# --------------------------------------------------

OPEN_VALUE = 0.045
CLOSE_VALUE = 0.0

gripper_open = True
running = True


# --------------------------------------------------
# Keyboard controls
# --------------------------------------------------

def key_callback(keycode):
    global gripper_open, running

    try:
        key = chr(keycode).lower()
    except ValueError:
        return

    # O = Open gripper
    if key == "o":
        gripper_open = True
        print("Gripper: OPEN")

    # C = Close gripper
    elif key == "c":
        gripper_open = False
        print("Gripper: CLOSE")

    # Q = Quit
    elif key == "q":
        running = False
        print("Program: QUIT")


# --------------------------------------------------
# Start MuJoCo viewer
# --------------------------------------------------

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    print("--------------------------------")
    print("UR3e Gripper Controller")
    print("--------------------------------")
    print("O = Open gripper")
    print("C = Close gripper")
    print("Q = Quit")
    print("--------------------------------")

    while viewer.is_running() and running:

        # ------------------------------------------
        # Control the gripper
        # ------------------------------------------

        if gripper_open:
            data.ctrl[6] = OPEN_VALUE
            data.ctrl[7] = OPEN_VALUE

        else:
            data.ctrl[6] = CLOSE_VALUE
            data.ctrl[7] = CLOSE_VALUE

        # ------------------------------------------
        # Advance simulation
        # ------------------------------------------

        mujoco.mj_step(model, data)

        # Update viewer
        viewer.sync()

        # Small delay
        time.sleep(0.002)

print("MuJoCo program stopped.")