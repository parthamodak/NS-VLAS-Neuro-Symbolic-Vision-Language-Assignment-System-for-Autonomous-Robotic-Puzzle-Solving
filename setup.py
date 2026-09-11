from glob import glob

from setuptools import find_packages, setup


package_name = "ur3e_mujoco_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/ur3e_mujoco_bridge"]),
        (f"share/{package_name}", ["package.xml", "local_setup.dsv"]),
        (f"share/{package_name}/hook", ["hook/ament_prefix_path.dsv"]),
        (f"share/{package_name}/environment", ["environment/ament_prefix_path.dsv"]),
        (f"share/{package_name}/simulation", ["simulation/scene.xml", "simulation/LICENSE.txt"]),
        (f"share/{package_name}/simulation/assets", glob("simulation/assets/*.obj")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mmpar",
    maintainer_email="mmpar@example.com",
    description="Low-memory ROS 2 bridge for an existing UR3e MuJoCo scene.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mujoco_bridge = ur3e_mujoco_bridge.mujoco_bridge:main",
        ],
    },
)
