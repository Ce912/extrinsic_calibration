#!/usr/bin/env python
import cv2
import numpy as np
import rospy
import sys
#import moveit_commander
import pyrealsense2 as rs
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
z_offset = 0.0 #m

#Get intrinsic and extrinsic camera parameters 
intrinsic_file =("./calibration_results/intrinsic_rs.yaml")
extrinsic_file =("./calibration_results/extrinsic_rs.yaml")

with open(intrinsic_file, "r") as f:
    data = yaml.safe_load(f)

with open(extrinsic_file, "r") as f:
    extrinsic = yaml.safe_load(f)
    
camera_matrix = np.array(data["matrix"], dtype=np.float32)
dist_coeff = np.array(data["distorion"], dtype=np.float32)
#print(camera_matrix, dist_coeff)
#input('Check camera matrix and distortion coeff shape')

#T_camera_to_base = np.array(extrinsic["Extrinsic matrix"], dtype =np.float64)
#print(T_camera_to_base)
#input('Check extrinsic matrix shape')


#Realsense D405
"""fx=396.70410057
fy=395.99342059
cx=319.39414917
cy= 243.79609093"""


"""fx=397.53244523
fy=396.30969828
cx=310.28975208
cy= 243.70406886

#dist_coeff = np.array([-5.71410040e-02, 7.55566235e-02 ,   5.22313193e-05 ,  3.93650715e-03,  -4.76807683e-02], dtype=np.float32)

dist_coeff = np.array([-0.0734865,   0.1103979,   0.00147756, -0.00054896, -0.10140231], dtype=np.float32)

camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)"""


T_camera_to_base= np.array([[ 0.00414945,  0.78021702, -0.62549515 , 0.5992],
 [ 0.99932014 , 0.01967936,  0.03117658, -0.0806],
 [ 0.03663384, -0.62519927 ,-0.77960492 , 0.4387],
 [ 0,        0,          0,          1      ]])


""" 
T_camera_to_base= np.array([[ 0.04879571,  0.77299747 ,-0.63252976 , 0.63329932],
 [ 0.99735475, -0.00355027,  0.07260098 ,-0.0763331 ,],
 [ 0.05387472, -0.63439917, -0.77112593 , 0.4644947 ],
 [ 0,         0,          0,         1,        ]], dtype=np.float64)

print(T_camera_to_base) """



""""
T2_camera_to_base= np.array([[ 0.0548203,   0.74981983, -0.65936709,  0.61764738],
 [ 0.99521962, -0.09448597, -0.02470433, -0.18922425],
 [-0.08082474, -0.65486077, -0.75141515,  0.46043474],
 [ 0,         0,          0,         1,        ]], dtype =np.float64)"""


# ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50) 
aruco_params = cv2.aruco.DetectorParameters()

# The size of your marker in meters
marker_length = 0.173  # Example: 17.5 cm marker

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
    
    current_pose = group.get_current_pose().pose
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


    group.set_pose_target(target_pose)

    
    
    # For orientation (if needed):
    # quaternion = rot.as_quat()  # Use this if you want to also set orientation
    # group.set_pose_target([position[0], position[1], position[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    
    group.go(wait=True)
    group.stop()
    group.clear_pose_targets()


    #display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    #display_trajectory.trajectory_start = robot.get_current_state()
    #display_trajectory.trajectory.append(plan)
    # Publish
    #display_trajectory_publisher.publish(display_trajectory)

def main():
    # Assuming you are capturing frames from a camera (e.g., Realsense)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    #pipeline.start(config)

    pipeline.start(config)
    
    
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
            #corners, ids, _ = cv2.aruco.ArucoDetector.detectMarkers(gray)
            
            if ids is not None:
                # Process each detected marker
                for marker_corners in corners:
                    # Use solvePnP to estimate the marker's pose
                    # marker_corners[0] gives the 2D image coordinates of the 4 corners
                    matrix= np.array([[433.437,0, 429.824],  
                             [  0        , 432.911, 241.947],
                                [  0        ,   0        ,   1       ]])
                    
                    dist= np.array([[-0.0531649,0.0617309,0.000420005 ,0.000574532,-0.0206657]])

                    #et, rvec, tvec = cv2.solvePnP(object_points, marker_corners[0], camera_matrix, dist_coeff)
                    ret, rvec, tvec = cv2.solvePnP(object_points, marker_corners[0], matrix, dist)
                    intrinsic = color_frame.profile.as_video_stream_profile().intrinsics
                    
                    #print(intrinsic)
                    
                    #ret, rvec, tvec = cv2.solvePnP(object_points, marker_corners[0])
                    
                    if ret:
                        # Draw the marker axes for visualization
                        #v2.drawFrameAxes(color_image, camera_matrix, dist_coeff, rvec, tvec, 0.05)
                        cv2.drawFrameAxes(color_image, matrix, dist, rvec, tvec, 0.05)
                        pose_in_base, rot_in_base = transform_camera_to_base(tvec, rvec)
                        print(f"The marker center pose is: {pose_in_base}  " )

                    
                        

                        #print the output rotation vector
                        rot_ = R.from_matrix(rot_in_base)
                        rot_2,_ = cv2.Rodrigues(rot_in_base)
                        rot_vector = rot_.as_rotvec()  #is a matrix
                        rot_euler = rot_.as_euler('zxy', degrees=True)

                        #print(rot_vector)
                        #input('rotation as rotvect')
                        print(f"The marker orientation is: {rot_euler}  " )
                        #input('rotation as euler angles')
                        #print(rot_2)
                        #input('rotation2')
                        #print(pose_in_base)Fim
                        
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
        
