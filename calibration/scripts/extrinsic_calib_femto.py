#!/usr/bin/env python
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2, os
from cv2 import aruco
import rospy 
import tf2_ros
import glob
import pyorbbecsdk as femto 
from pyorbbecsdk import OBSensorType, OBFormat, VideoStreamProfile, FrameSet, OBError
import yaml
#from utils import frame_to_bgr_image
import geometry_msgs as geom_msg
from std_msgs.msg import Bool
import moveit_msgs.msg 
import tf


#Extrinsic calibration code exploiting MoveIt and openCv. Final results: camera to robot's base transformation.
#The robot is teleoperated to N different poses to acquire images of the ChArUcO marker mounted on the eof.
#Discard or save images, selecting yes/no. 
#CAVEAT: Toggle "Collision Aware IK" and "Cartesian Path" on the Rviz interface

#Save images in a local folder


rel_path = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(rel_path, '../extrinsic_femto_images/*.png')

images = glob.glob(target_path)

#Get CharucoBoard parameters
board_rows = rospy.get_param("~board_rows", 8)
board_columns = rospy.get_param("~board_columns", 11)
square_length = rospy.get_param("~square_length", 0.015)
marker_length = rospy.get_param("~marker_length", 0.011)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
board = cv2.aruco.CharucoBoard((board_columns, board_rows), square_length, marker_length, aruco_dict)
board.setLegacyPattern(True)

#Transformation matrices initialization
T_b2ee = np.eye(4)
T_B2EE = []
rot_B2EE = []
tvect_B2EE = []
T_C2B = np.eye(4)
T_t2c = np.eye(4)
T_t2C = []
rot_t2C = []
tvect_t2C = []

#Counter initialization
pose_counter = 0

#Get Max poses parameters
max_poses = rospy.get_param("~max_poses", 10)

#Create directory for results saving
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


#Get charuco board object points for detection
def get_charuco_board_object_points (board, corners_ids: list | np.ndarray):
    corners = board.getChessboardCorners()
    object_points = []
    for idx in corners_ids:
        object_points.append(corners[idx])
    return np.array(object_points, dtype=np.float32)


#MoveIt interface communication
def calibration(folder_path, pose_callback):

    # Create directory to save images
    create_directory(folder_path)
     
    # Subscribe to the /terminate topic
    rospy.Subscriber('/execute_trajectory/feedback', moveit_msgs.msg.ExecuteTrajectoryActionFeedback, pose_callback, folder_path)
    rospy.spin()
    # Keep the node running

#Camera stream handling
def image_streaming():
    width = rospy.get_param("~camera_res_w", 1280)
    height = rospy.get_param("~camera_res_h", 720)

    # Initialize the FemtoBolt camera pipeline
    pipeline = femto.Pipeline()
    config = femto.Config()

    # Configure the pipeline to stream in color
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        #rospy.loginfo(profile_list)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(width,height, OBFormat.RGB, 30)

        except OBError as e:
            rospy.loginfo(e)
            color_profile = profile_list.get_default_video_stream_profile()
            rospy.loginfo("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        rospy.loginfo(e)
        return
    
    camera_param = pipeline.get_camera_param()
    intr = camera_param.rgb_intrinsic
    fx = intr.fx
    fy = intr.fy 
    cx = intr.ppx
    cy = intr.ppy 

    dist = np.array(intr.coeffs)
    matrix = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,0,1]])


    pipeline.start(config)
    try:
        while True: 
            rospy.sleep(2) 
            frames: FrameSet = pipeline.wait_for_frames(100)
            # Wait for a coherent color frame
            if frames is None:
                continue
            color_frame = frames.get_color_frame()

            # Convert to numpy array
            col_image = np.asanyarray(color_frame.get_data())
            reshaped_image = col_image.reshape((height, width, 3))
            color_image= cv2.cvtColor(reshaped_image, cv2.COLOR_BGR2RGB)

            if not color_frame:
                rospy.logwarn("No color frame captured.")
                return             
                            
            # Save the image
            timestamp = rospy.get_time()  # Get timestamp for unique naming
            image_name = os.path.join(folder_path, f"image_{timestamp}.png")
            cv2.imwrite(image_name, color_image)
            rospy.loginfo(f"Image saved at {image_name}")

            # Stop the camera pipeline
            return image_name
    except KeyboardInterrupt:
        rospy.loginfo("Calibration stopped by the user")
    pipeline.stop()
    return image_name, matrix, dist

#Board detection: it outputs the Tt2C (target2camera) transform matrix
def image_processing(image_name, frame, matrix, dist, board, verbose=True):
    
    rospy.loginfo("=> Processing image {0}".format(image_name))
    rospy.loginfo("POSE ESTIMATION STARTS:")

    #Define charuco board parameters 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    frame = cv2.imread(image_name)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    detector_params = cv2.aruco.CharucoParameters()
    detector_params.minMarkers = 2
    detector_params.tryRefineMarkers = True

    #Set intrinsic camera parameters
    detector_params.cameraMatrix = matrix
    detector_params.distCoeffs = dist
    charucodetector = cv2.aruco.CharucoDetector(board, detector_params)
    charucodetector.setBoard(board)

    charuco_corners, charuco_ids, marker_corners, marker_ids = charucodetector.detectBoard(gray)

    #Error handling for no detection       
    if marker_corners is None or marker_ids is None:
        rospy.loginfo(f'No detection, try again')
        return None
    if len(marker_corners) != len(marker_ids) or len(marker_corners) == 0:
        rospy.loginfo(f'Wrong detection, try again')
        return None
    
    #Draw detected markers
    cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

    try:
          
        object_points = get_charuco_board_object_points(board, charuco_ids)
        charuco_corners2 = cv2.cornerSubPix(gray, charuco_corners, (23,23), (-1,-1), criteria)
        ret, p_rvec, p_tvec_t2c = cv2.solvePnP(object_points, charuco_corners2, matrix,dist)  #in camera frame (2d Image plane)

        if p_rvec is None or p_tvec_t2c is None:
            return None
        if np.isnan(p_rvec).any() or np.isnan(p_tvec_t2c).any():
            return None
        img_axes = cv2.drawFrameAxes(frame,
                            matrix,
                            dist,
                            p_rvec,
                            p_tvec_t2c,
                            0.1)

    except cv2.error:
        return None
        

    #convert p_rvec from SolvePnP with Rodrigues to a rotation matrix
    rot_t2c, _ = cv2.Rodrigues(p_rvec)  #3x3
    
    T_t2c = np.eye(4)  #target to camera transform
    T_t2c[:3, :3] = rot_t2c
    T_t2c[:3, 3] = p_tvec_t2c.T  

    if verbose:
        rospy.loginfo('Translation : {0}'.format(p_tvec_t2c))
        rospy.loginfo('Rotation    : {0}'.format(p_rvec))
        rospy.loginfo('Distance from camera: {0} m'.format(np.linalg.norm(p_tvec_t2c)))

    return T_t2c, rot_t2c, p_tvec_t2c, img_axes  


#Show image of the grid and wait for user prompt. Compute TB2EE and Tt2C matrices and store them
def pose_callback(feedback_msg, image_name, max_poses, matrix, dist):
    global pose_counter, T_B2EE, rot_B2EE, tvect_B2EE, T_t2C, rot_t2C, tvect_t2C

    if feedback_msg.feedback.state == "IDLE":  # If the /terminate signal is True
            image_name = image_streaming()
            frame = cv2.imread(image_name)
            cv2.imshow('img',frame)
            cv2.waitKey(500)

            user_input = input("Do you want to save the image? (yes/no): ")

            if user_input.lower() in ["yes", "y"]:           
                rospy.loginfo("Terminate signal received. Waiting for 2 seconds before capturing image...")
                rospy.sleep(2)  # Wait for 2 seconds

                try: 
                    
                    tfBuffer = tf2_ros.Buffer()
                    rospy.sleep(4)
            
                    #Get the transform from base reference frame to gripper reference frame
                    res = tfBuffer.lookup_transform("panda_EE", "panda_link0",rospy.Time(0))
                    tvect_b2ee = np.array([res.transform.translation.x,res.transform.translation.y, res.transform.translation.z]) 
                    rot = np.array([res.transform.rotation.x,res.transform.rotation.y, res.transform.rotation.z, res.transform.rotation.w])
                    r = R.from_quat(rot)
                    rot_b2ee = r.as_matrix()
                    
                    #assembly the transformation matrix
                    T_b2ee[:3, :3] = rot_b2ee
                    T_b2ee[:3, 3] = tvect_b2ee.T 
                    rospy.loginfo(T_b2ee)
                    rospy.loginfo(f'b2ee')

                except Exception as e:
                    rospy.loginfo(f"Transform Error: {e}")
                
                #Capture and process the image
                if image_name:
                    frame = cv2.imread(image_name)
                    T_t2c, rot_t2c, tvect_t2c, img_axes = image_processing(image_name,frame, matrix, dist, board, verbose =True)
                    cv2.imshow('img with axes',img_axes)
                    cv2.waitKey(500)

                    #rot_t2c is a multidimensional rotation matrix (3x3,), tvect_t2c is a cmultidimensional column vector (3,) 
                    # T_t2c is a 4x4 trasformation matrix

                    if T_t2c is not None:
                        tvect_t2c = tvect_t2c.flatten() 
                        T_t2C.append(T_t2c)
                        rot_t2C.append(rot_t2c) #3x3
                        tvect_t2C.append(tvect_t2c)

                        #Convert to array
                        rot_t2C_mat = np.array(rot_t2C) #column array of 3x3 rotation matrices--> check and eventually transpose to have matrices next to each other
                        tvect_t2C_mat = np.array(tvect_t2C) # matrix of column translation vectors stack one next to each other
    
                        #Store the matrix in an array
                        T_B2EE.append(T_b2ee)
                        rot_B2EE.append(rot_b2ee)  
                        tvect_B2EE.append(tvect_b2ee)  
                        
                        #Convert from list to array
                        rot_B2EE_mat = np.array(rot_B2EE) 
                        tvect_B2EE_mat = np.array(tvect_B2EE)
                    pose_counter += 1
                    rospy.loginfo(f"Pose {pose_counter} acquired successfully")

                    if pose_counter >= max_poses: 
                        #Compute rotation matrix and translation vector
                        rot_c2b, t_vect_c2b = cv2.calibrateHandEye(rot_B2EE_mat,tvect_B2EE_mat,rot_t2C_mat, tvect_t2C_mat, cv2.CALIB_HAND_EYE_PARK)
                        
                        #Assembly the transformation matrix
                        T_C2B[:3, :3] = rot_c2b
                        T_C2B[:3, 3] = t_vect_c2b.T   

                        rospy.loginfo("Final Transformation Matrix (Camera to robot base):")
                        rospy.loginfo(T_C2B)
                        return T_C2B
            else:
                rospy.loginfo("Change robot pose and try again")

#Save results in a yaml file
                            
#Absolute path 
#saving_directory = os.path.join(rel_path, '../extr_calib_results/')
#create_directory(saving_directory)
#saving_path = os.path.join(saving_directory, "extrinsic_femto.yaml")

#Parametetric absolute path
output_dir = rospy.get_param("~output_file")
output_file = os.path.join(rel_path, output_dir)
create_directory(output_file)


#Organize data for saving in yaml
TC2B = T_C2B.tolist()
data = {
    'Camera' : "FemtoBolt", 
    'Extrinsic matrix': TC2B,
    'Poses acquired' : max_poses,
}

with open(output_file, "w") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

rospy.loginfo(f'Results saved in ', output_file)

if __name__ == '__main__':
    rospy.init_node('calibrator', anonymous=False)
    try:
        folder_path = target_path
        calibration(folder_path, pose_callback)

    except rospy.ROSInterruptException:
        pass
