# extrinsic_calibration
This repo provides a useful ready-to-use package for extrinsic camera calibration, exploiting OpenCv, MoveIt! and tf library. The examples refer to RealSense and FemtoBolt cameras, but they can be easily adapted.
In the calibration folder, you can find the scripts for: 
- extrinsic camera calibration
- marker detection for verifying the results

The ChArUco grid is used as fiducial marker for the extrinsic calibration, while a single ArUco marker is used for results verification.
The procedure requires a Franka Panda robot and the relative computer interface. 

The intrinsic folder contains the python scripts for intrinsic calibration. Please refer to the OpenCv website for more detailed information. 
For intrinsic calibration, a chessboard is exploited as fiducial marker. 

# Guidelines: 
Before calibration, set the relevant parameters in the calibration/config/ folder to adapt them to your setup (i.e. grid and 
To execute the program you'll need to launch:
> roslaunch panda_moveit_config franka_control.launch robot_ip:<ip_robot>

Toggle "Cartesian Path" and "Collision aware IK" in the launched Rviz interface 

Other terminal: 
> roslaunch calibration extrinsic_rs_calib.launch

The results will be stored in a local yaml file. 
To verify the results, you can use the verification scripts. For the RealSense (e.g.): 
>rosrun calibration verification_rs.py

