#!/usr/bin/env python
import cv2
import numpy as np
import rospy
import sys
import moveit_commander
from freenect2 import Device, FrameType, Frame, ColorCameraParams, IrCameraParams
from scipy.spatial.transform import Rotation as R
import moveit_msgs.msg
import geometry_msgs.msg
from numpy.linalg import inv
import tf2_ros


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
   

color_fx = 1.0449799595954407e+03
color_fy = 1.0438765671806525e+03
color_cx = 9.6811739605485991e+02
color_cy = 5.7587106255468348e+02
color_k1 = 1.2534658613104072e-02
color_k2 = 3.1411494891863845e-02
color_p1 = 2.3510200869053225e-03
color_p2 = 2.8106139131822114e-04
color_k3 = -8.5274215921585836e-02 


#IR_parameters 
ir_fx = 3.6138964895211541e+02
ir_fy = 3.6002111699341555e+02
ir_cx = 2.5742689375766264e+02
ir_cy = 2.0853278095439734e+02
ir_k1 = -3.3161494601389508e-03
ir_k2 = -5.0573788168162104e-02
ir_p1 = 5.6394712918869340e-03
ir_p2 = -5.2760669597451126e-03
ir_k3 = 0.2050978842362527


color_camera_matrix = np.array([[color_fx, 0, color_cx],
                          [0, color_fy, color_cy],
                          [0, 0, 1]], dtype=np.float32)
color_dist_coeff = np.array([color_k1, color_k2,   color_p1 , color_p2,  color_k3], dtype=np.float32)

ir_camera_matrix = np.array([[ir_fx, 0, ir_cx],
                          [0, ir_fy, ir_cy],
                          [0, 0, 1]], dtype=np.float32)
ir_dist_coeff = np.array([ir_k1, ir_k2,   ir_p1 , ir_p2,  ir_k3], dtype=np.float32)




"""

T_camera_to_base =np.array([[-0.62787736, -0.3055215,  -0.71583981, 0.50081483],
 [ 0.77177245, -0.12541953, -0.62340775,  0.66799249],
 [ 0.10068418, -0.94388906, 0.31454116, -0.01176117],
 [ 0,       0,         0,          1,       ]], dtype=np.float64)"""


T_camera_to_base = np.array([[-0.79686169, -0.27073203, -0.54010705 , 0.38364225],
 [ 0.59284759, -0.17819575, -0.78535215,  0.69835839],
 [ 0.1163752,  -0.9460182 ,  0.3025002,  0.05044301],
 [ 0,          0,          0,         1,        ]], dtype= np.float64)   #best results 

"""T_camera_to_base = np.array([[-0.59730141, -0.25317444 ,-0.76100836,  0.51388956],
 [ 0.77183524 , 0.07639913, -0.63121591 , 0.75876023],
 [ 0.21794811, -0.96439923 , 0.14977564 , 0.16213553],
 [ 0,        0,         0,         1,        ]], dtype=np.float64)"""



"""
R_base_to_camera = T_base_to_camera[:3,:3]
t_base_to_camera = T_base_to_camera[:3,3]
R_camera_to_base = R_base_to_camera.T  #inverse rotation
t_camera_to_base = np.matmul(-R_camera_to_base, t_base_to_camera)
print(t_camera_to_base)
print(f'size is : {t_camera_to_base.shape}')
input('t_camera_to_base')
T_camera_to_base = np.eye(4)
T_camera_to_base[:3,:3] = R_camera_to_base
T_camera_to_base[:3, 3] = t_camera_to_base"""



#base to camera : the origin is in the camera position, with the reference frame aligned with the robot base. It must be traslated


#T_camera_to_base= np.linalg.inv(T_base_to_camera)


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
    #input('T marker to camera')
    #print(T_marker_to_camera_new)
    #input('T marker to camera new')  #same as t marker camera
    #T_marker_to_camera[:3,3] = tvec

    #print(tvec)
    #input('tvec')
    #print(rotation_matrix)
    #input('rotation_matrix')
    
    T_marker_in_base = np.dot(T_camera_to_base, T_marker_to_camera) 

    #T_marker_in_base_new = np.matmul(T_camera_to_base, T_marker_to_camera)  #scalar product between two matrices (a,b) as a-scalar-b. Proper order
    #T_marker_in_base_new2 = np.matmul( T_marker_to_camera,T_camera_to_base)  #

    #print(T_marker_in_base)
    #input('np.dot result')
    #print(T_marker_in_base_new)
    #input('matmul result')

    #T_marker_in_base =  T_camera_to_base @ T_marker_to_camera  #asse z verso il basso
    #t_new =  np.dot( T_marker_in_base, T_camera_to_base)
    

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

def get_next_frames(device):
    # These are the types of the frames we want to capture and the order
    # they should be returned.
    required_types = [FrameType.Color, FrameType.Depth, FrameType.Ir]
    
    # Store incoming frame in this dictionary keyed by type.
    frames_by_type = {}
    
    for frame_type, frame in device:
        # Record frame
        frames_by_type[frame_type] = frame
        
        # Try to return a frame for each type. If we get a KeyError, we need to keep capturing
        try:
            return [frames_by_type[t] for t in required_types]
        except KeyError:
            pass # This is OK, capture the next frame


def main():
    # Assuming you are capturing frames from a camera (e.g., Realsense)

    device = Device()
    #device.open()
    
    device.color_camera_params = ColorCameraParams() #camera color intrinsic calibration parameters
    device.color_camera_params.fx = color_fx
    device.color_camera_params.fy = color_fy
    device.color_camera_params.cx = color_cx
    device.color_camera_params.cy = color_cy

    device.ir_camera_params = IrCameraParams() #camera IR intrinsic calibration parameters
    device.ir_camera_params.fx = ir_fx #to change with IR parameters after intrinsic calibration
    device.ir_camera_params.fy = ir_fy
    device.ir_camera_params.cx = ir_cx
    device.ir_camera_params.cy = ir_cy
    device.ir_camera_params.k1 = ir_k1
    device.ir_camera_params.k2 = ir_k2
    device.ir_camera_params.k3 = ir_k3
    device.ir_camera_params.p1 = ir_p1
    device.ir_camera_params.p2 = ir_p2
    
    #color_ = Frame(frame_ref=Frame.color)
    
    
    
    with device.running():
        while not rospy.is_shutdown():
            color, depth, ir = get_next_frames(device)
            #listener = device.color_frame_listener(FrameType.Color, Frame.bytes_per_pixel)
            
            #rospy.loginfo('listener')
            #device.start(frame_listener=listener)
            #rospy.loginfo('list')
            
            #frames_obj = device.device.close()get_next_frame(timeout=2)
            #print(frames_obj)
            #print(dir(frames))
            #rospy.loginfo('frame obj')
            #frames= frames_obj[1]
            #print(frames)
            #print(dir(frames))
            #rospy.loginfo('frames')
            #color_ = frames.getColorFrame()
            
            #rospy.loginfo('col_frame')
            

            
            if color:
                color_frame = color.to_array().astype(np.uint8)
                #print(type(color_frame))
                #print(color_frame.shape)
                #input('check color frame type')
                

                #cv2.imshow('rgb', color_frame)
                #cv2.imshow(100)
                #device.close()

                    
                # Configure the pipeline to stream in color
                #config.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)  # Example resolution and format  try to increase resolution up to 1920, 1080
                                
                # Start streaming
                #pipeline.start(config)
                                        
                # Wait for a coherent color frame
                #frames = pipeline.wait_for_frames()
                
                                
                if type(color_frame) == None:
                    rospy.logwarn("No color frame captured.")
                    return
                
                                
                # Convert to numpy array
                #color_image = np.asanyarray(color_frame.get_data())

                #Kinect
                #color_image = color_frame 
                color_frame = np.flip(color_frame,1)
                #print(color_frame)
                rospy.loginfo("color image")
                

                
                # Convert to numpy array
                color_image = color_frame
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                #rospy.sleep(5)
                
                # Detect ArUco markers
                detector = cv2.aruco.ArucoDetector(aruco_dict)
                corners, ids, _ = detector.detectMarkers(gray)
                #corners, ids, _ = cv2.aruco.ArucoDetector.detectMarkers(gray)
                
                if ids is not None:
                    # Process each detected marker
                    for marker_corners in corners:
                        # Use solvePnP to estimate the marker's pose
                        # marker_corners[0] gives the 2D image coordinates of the 4 corners
                        ret, rvec, tvec = cv2.solvePnP(object_points, marker_corners[0], color_camera_matrix, color_dist_coeff)
                        
                        if ret:
                            # Draw the marker axes for visualization
                            image_with_axes=cv2.drawFrameAxes(color, color_camera_matrix, color_dist_coeff, rvec, tvec, 0.05)

                            pose_in_base, rot_in_base = transform_camera_to_base(tvec, rvec)
                            rospy.loginfo(pose_in_base)
                        
                            

                            #print the output rotation vector
                            rot_ = R.from_matrix(rot_in_base)
                            rot_2,_ = cv2.Rodrigues(rot_in_base)
                            rot_vector = rot_.as_rotvec()  #is a matrix
                            rot_euler = rot_.as_euler('zxy', degrees=True)

                            #print(rot_vector)
                            #input('rotation as rotvect')
                            print(rot_euler)
                            #input('rotation as euler angles')
                            #print(rot_2)
                            #input('rotation2')
                            #print(pose_in_base)
                            
                            # Move the robot gripper to the detected marker position
                            #move_robot_to_pose(pose_in_base, rot_in_base, z_offset)
                            
                        # Display the image
                cv2.imshow('Aruco Detection', image_with_axes)
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else :
                rospy.logerr(" no detection")
    #device.close()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass
        
