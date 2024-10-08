from setuptools import find_packages, setup
from glob import glob
import os

package_name = "dynamics_simulation"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yashas",
    maintainer_email="yashas.amb@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "simulator_node = dynamics_simulation.simulator_node:main",
            "ackermann_pub = dynamics_simulation.ackermann_pub:main",
        ],
    },
)
