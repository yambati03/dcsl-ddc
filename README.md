## Dynamic Drift Control

`scripts/optimize_tire_model.py` finds the optimal parameters for the [Pacejka model](https://www.jstor.org/stable/44470677) to minimize error against experimental data.

### Model Validation

The `validate_model.py` script plots the trajectory from a given log against the trajectory predicted by the optimized dynamics model. 

* The script takes in the filepath of the ROS bag to run as a command line argument.
* The predicted trajectory (green) is plotted if the `-p` flag is specified. It is _not_ plotted by default.
* Plots of the heading, yaw rate, and velocity are shown prior to the simulation if the `-s` flag is specified.

#### Steps to run `validate_model.py`

1. Since the script loads a ROS bag that depends on custom message types (`AckermannDriveStamped`, `ViconObject`), the script must be run within a ROS workspace with the packages that define these types. First, create a ROS workspace by running `mkdir -p ros_ws/src`.
2. Run `cd ros_ws/src` and clone this repository. Then clone the `ackermann_msgs` repo by running `git clone -b ros2 https://github.com/ros-drivers/ackermann_msgs.git`, and the `vicon_ros` repo by running `git clone https://github.com/yambati03/vicon_ros.git`
3. Build the workspace by running `colcon build --symlink-install` in the top-level `ros_ws` directory. Then, source the install directory by running `. install/setup.bash`
4. Then, navigate to `dcsl-ddc/scripts`. The `validate_model.py` script can be run using `python3 validate_model.py /path/to/bag.db3 -p -s`.