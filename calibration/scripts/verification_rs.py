#!/usr/bin/env python
import cv2
import numpy as np
import rospy
import sys
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import tf2_ros
import yaml
import os

#Verify extrinsic parameters by detecting aruco marker. The output is the position of the marker center

# Initialize ROS node
rospy.init_node('aruco_pose_estimator', anonymous=True)

#Get extrinsic camera parameters 
rel_path = os.path.dirname(os.path.abspath(__file__))
extrinsic_file = os.path.join(rel_path,"../extr_calib_results/extrinsic_rs.yaml")

with open(extrinsic_file, "r") as f:
    extrinsic = yaml.safe_load(f)
T_camera_to_base= np.array(extrinsic["TC2B"], dtype=np.float32)

#ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50) 
aruco_params = cv2.aruco.DetectorParameters()

# The size of your marker in meters
marker_length = 0.173  # Example: 17.5 cm marker


# Corners of the marker in the marker's local coordinate frame
object_points = np.array([
    [-marker_length / 2, marker_length / 2, 0],  # Top-left corner
    [marker_length / 2, marker_length / 2, 0],   # Top-right corner
    [marker_length / 2, -marker_length / 2, 0],  # Bottom-right corner
    [-marker_length / 2, -marker_length / 2, 0]  # Bottom-left corner
], dtype=np.float32)


#Get t2c (target to camera) transformation components
def transform_camera_to_base(tvec,rvec):

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    T_marker_to_camera = np.eye(4)
    T_marker_to_camera[:3,:3] = rotation_matrix
    T_marker_to_camera[:3,3] = tvec.flatten()

    T_marker_in_base = np.dot(T_camera_to_base, T_marker_to_camera) 
    pose_in_base = T_marker_in_base[:3,3]
    rot_in_base = T_marker_in_base[:3,:3]
    return pose_in_base, rot_in_base

def main():
    #Start camera pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    #Set resolution according to intrinsic parameters 
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    cfg =  pipeline.start(config)

    profile = cfg.get_stream(rs.stream.color) # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics()
    #Get actual intrinsic camera parameters
    fx = intr.fx
    fy = intr.fy 
    cx = intr.ppx
    cy = intr.ppy 

    dist = np.array(intr.coeffs)
    matrix = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,0,1]])

    try:
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames() 
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
        
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers
            detector = cv2.aruco.ArucoDetector(aruco_dict)
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None:
                # Process each detected marker
                for marker_corners in corners:
                    # Use solvePnP to estimate the marker's pose
                    ret, rvec, tvec = cv2.solvePnP(object_points, marker_corners[0], matrix, dist)
     
                    if ret:
                        # Draw the marker axes for visualization
                        cv2.drawFrameAxes(color_image, matrix, dist, rvec, tvec, 0.05)
                        pose_in_base, rot_in_base = transform_camera_to_base(tvec, rvec)
                        rospy.loginfo(f"The marker center pose is: {pose_in_base}  " )

                        rot_ = R.from_matrix(rot_in_base)
                        rot_euler = rot_.as_euler('zxy', degrees=True)

                        rospy.loginfo(f"The marker orientation is: {rot_euler}  " )
                        
            # Display the image
            cv2.imshow('Aruco Detection', color_image)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass
        
