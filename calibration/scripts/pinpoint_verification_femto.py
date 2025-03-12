#!/usr/bin/env python
import cv2
import numpy as np
import rospy
import sys
#import moveit_commander
import pyorbbecsdk as femto 
from pyorbbecsdk import OBSensorType, OBFormat, FrameSet, VideoStreamProfile, OBError
from scipy.spatial.transform import Rotation as R
import moveit_msgs.msg
import geometry_msgs.msg
from numpy.linalg import inv
import tf2_ros
import yaml


# Initialize moveit_commander for controlling the robot arm
#moveit_commander.roscpp_initialize(sys.argv)

# Initialize ROS node
rospy.init_node('aruco_pose_estimator', anonymous=True)
#robot = moveit_commander.RobotCommander()
group_name = "panda_arm"
#group = moveit_commander.MoveGroupCommander(group_name)

#scene = moveit_commander.PlanningSceneInterface()
display_trajectory_publisher = rospy.Publisher(
                                    '/move_group/display_planned_path',
                                    moveit_msgs.msg.DisplayTrajectory,  queue_size=20)

# The size of your marker in meters
marker_length = 0.175  # Example: 17.5 cm marker
z_offset = 0.0 #m

#Get intrinsic and extrinsic camera parameters 
intrinsic_file =("./calibration_results/intrinsic_femto.yaml")
extrinsic_file =("./calibration_results/extrinsic_femto.yaml")

with open(intrinsic_file, "r") as f:
    data = yaml.safe_load(f)

with open(extrinsic_file, "r") as f:
    extrinsic = yaml.safe_load(f)
camera_matrix = np.array(data["matrix"], dtype=np.float32)
dist_coeff = np.array(data["distorion"], dtype=np.float32)
print(camera_matrix, dist_coeff)
input('Check camera matrix and distortion coeff shape')

T_camera_to_base = np.array(extrinsic["Extrinsic matrix"], dtype =np.float64)
print(T_camera_to_base)
input('Check extrinsic matrix shape')

#FemtoBolt 
fx=742.96789955
fy=741.38497421
cx=640.16218029
cy= 363.71992169

dist_coeff = np.array([ 6.38110551e-02, -2.21760695e-01,  3.51841251e-03, -3.86512630e-04, 5.80424357e-01], dtype=np.float32)


camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)


T_camera_to_base= np.array([[-8.36532831e-01,  5.27823111e-02, -5.45368546e-01,  5.20650404e-01],
 [ 5.44244076e-01, -3.50078340e-02, -8.38196181e-01,  5.67754928e-01],
 [-6.33341031e-02, -9.97992224e-01,  5.58742298e-04 , 1.11006932e-01],
 [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]], dtype =np.float64)


# ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50) 
aruco_params = cv2.aruco.DetectorParameters()

# 3D object points for an ArUco marker (assuming Z=0 for all points)
# These are the corners of the marker in the marker's local coordinate frame
object_points = np.array([
    [-marker_length / 2, marker_length / 2, 0],  # Top-left corner
    [marker_length / 2, marker_length / 2, 0],   # Top-right corner
    [marker_length / 2, -marker_length / 2, 0],  # Bottom-right corner
    [-marker_length / 2, -marker_length / 2, 0]  # Bottom-left corner
], dtype=np.float32)


def transform_camera_to_base(tvec,rvec):

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    T_marker_to_camera = np.eye(4)
    T_marker_to_camera[:3,:3] = rotation_matrix
    T_marker_to_camera[:3,3] = tvec.flatten()

    T_marker_to_camera_new = np.array([[rotation_matrix[0,0], rotation_matrix[0,1], rotation_matrix[0,2], tvec[0,0]],
                          [rotation_matrix[1,0], rotation_matrix[1,1], rotation_matrix[1,2], tvec[1,0]],
                          [rotation_matrix[2,0], rotation_matrix[2,1], rotation_matrix[2,2], tvec[2,0]],
                          [      0,        0,        0,          1]],dtype=np.float64)
    print(T_marker_to_camera)
    
    T_marker_in_base = np.dot(T_camera_to_base, T_marker_to_camera) 
    pose_in_base = T_marker_in_base[:3,3]
    #pose_in_base = t_new[:3,3]
    rot_in_base = T_marker_in_base[:3,:3]
    #print(pose_in_base)

    return pose_in_base, rot_in_base

def move_robot_to_pose(pose_in_base, rot_in_base, z_offset):

    """Move the robot's end effector to the estimated marker position with a Z-offset."""
    # Convert the rotation vector to a rotation matrix
    #rotation_matrix, _ = cv2.Rodrigues(rvec)
    rot = R.from_matrix(rot_in_base)

    # Extract translation (tvec)
    #position = tvec.flatten()
    pose_in_base = pose_in_base.flatten()
    #position = pose_in_base.T #row vector

    #pose is in the robot-base frame but is should be in the gripper frame
    #tfBuffer = tf2_ros.Buffer()
    #listener = tf2_ros.TransformListener(tfBuffer)
    #rospy.sleep(4)
    #res = tfBuffer.lookup_transform("panda_EE", "panda_link0",rospy.Time(0))

    #tvect_b2ee = np.array([res.transform.translation.x,res.transform.translation.y, res.transform.translation.z]) 
    #rot = np.array([res.transform.rotation.x,res.transform.rotation.y, res.transform.rotation.z, res.transform.rotation.w])


    #r = R.from_quat(rot)
    #rot_b2ee = r.as_matrix()
                    
    #T_b2ee=np.array([[rot_b2ee[0,0], rot_b2ee[0,1], rot_b2ee[0,2], tvect_b2ee[0] ],
                     #[rot_b2ee[1,0], rot_b2ee[1,1], rot_b2ee[1,2],tvect_b2ee[1] ],
                     #[rot_b2ee[2,0], rot_b2ee[2,1], rot_b2ee[2,2], tvect_b2ee[2]],
                     #[0,             0,                 0,          1]])
    #assembly the transformation matrix
       #check dimension

    #pose_in_base[2] += z_offset
    print(pose_in_base.shape)
    print(type(pose_in_base))

    
    
    pose = np.array([pose_in_base[0],
                     pose_in_base[1],
                     pose_in_base[2]+ z_offset,
                     1],dtype=np.float64)
    position = pose
    print(pose)
    input('check the height')
    
    

    #position = np.dot(T_b2ee, pose)
    #print(position)
    #rot_EE = np.dot(T_b2ee[:3,:3], rot_in_base)   #rotation matrix
    #print(rot_EE)
    #input('check')
    #rot_ =R.from_matrix(rot_EE)



    # Apply Z-offset for the gripper
    
    #current_pose = group.get_current_pose().pose
    # Move the robot to this position (ignoring orientation for now)
    pose_goal = geometry_msgs.msg.Pose()
    #pose_goal.header.frame_id = "panda_EE"
    pose_goal.position.x = position[0]
    pose_goal.position.y = position[1]
    pose_goal.position.z = position[2]
    pose_goal.orientation = current_pose.orientation
    

    #quaternion = rot.as_quat()
    #pose_goal.orientation.x = quaternion[0]
    #pose_goal.orientation.y = quaternion[1]
    #pose_goal.orientation.z = quaternion[2]
    #pose_goal.orientation.w = quaternion[3]

    


    # Set the target pose
    target_pose = geometry_msgs.msg.PoseStamped()
    target_pose.header.frame_id = "panda_link0"  # specify your planning frame
    target_pose.pose = pose_goal


    #group.set_pose_target(target_pose)

    # For orientation (if needed):
    # quaternion = rot.as_quat()  # Use this if you want to also set orientation
    # group.set_pose_target([position[0], position[1], position[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    #group.go(wait=True)
    #group.stop()
    #group.clear_pose_targets()
    #display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    #display_trajectory.trajectory_start = robot.get_current_state()
    #display_trajectory.trajectory.append(plan)
    # Publish
    #display_trajectory_publisher.publish(display_trajectory)

def main():
    # Assuming you are capturing frames from a camera (e.g., Realsense)
    pipeline = femto.Pipeline()
    config = femto.Config()

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        print(profile_list)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1280,720, OBFormat.RGB, 30)

        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
    pipeline.start(config)

    try:
        while True: 
            rospy.sleep(2) 
            frames: FrameSet = pipeline.wait_for_frames(100)
             
            # Wait for a coherent color frame
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            
            print(type(color_frame))
            #input()

            col_image = np.asanyarray(color_frame.get_data())
            reshaped_image = col_image.reshape((720, 1280, 3))
            color_image= cv2.cvtColor(reshaped_image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            if not color_frame:
                rospy.logwarn("No color frame captured.")
            
            # Detect ArUco markers
            detector = cv2.aruco.ArucoDetector(aruco_dict)
            corners, ids, _ = detector.detectMarkers(gray)
            #corners, ids, _ = cv2.aruco.ArucoDetector.detectMarkers(gray)
            
            if ids is not None:
                # Process each detected marker
                for marker_corners in corners:
                    # Use solvePnP to estimate the marker's pose
                    # marker_corners[0] gives the 2D image coordinates of the 4 corners
                    ret, rvec, tvec = cv2.solvePnP(object_points, marker_corners[0], camera_matrix, dist_coeff)
                    
                    if ret:
                        # Draw the marker axes for visualization
                        cv2.drawFrameAxes(color_image, camera_matrix, dist_coeff, rvec, tvec, 0.05)
                        pose_in_base, rot_in_base = transform_camera_to_base(tvec, rvec)
                        print(f"The marker center pose is: {pose_in_base}  " )

                        #Print the output rotation vector
                        rot_ = R.from_matrix(rot_in_base)
                        rot_2,_ = cv2.Rodrigues(rot_in_base)
                        rot_vector = rot_.as_rotvec()  #is a matrix
                        rot_euler = rot_.as_euler('zxy', degrees=True)

                        print(f"The marker orientation is: {rot_euler}  " )
                        
                        # Move the robot gripper to the detected marker position
                        #move_robot_to_pose(pose_in_base, rot_in_base, z_offset)
                        
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
        