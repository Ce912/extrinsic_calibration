#!/usr/bin/env python
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2, os
from cv2 import aruco
import rospy 
import tf2_ros
import glob
import pyrealsense2 as rs
import geometry_msgs as geom_msg
from std_msgs.msg import Bool
import moveit_msgs.msg 
import tf
import yaml

#Extrinsic calibration code exploiting MoveIt and openCv. Final results: camera to robot's base transformation.
#The robot is teleoperated to N different poses to acquire images of the ChArUcO marker mounted on the eof.
#Discard or save images, selecting yes/no. 
#CAVEAT: Toggle "Collision Aware IK" and "Cartesian Path" on the Rviz interface

#Save images in a local folder
images = glob.glob('./extrinsic_rs_images/*.png')

#Get CharucoBoard parameters
board_rows = rospy.get_param("~board_rows", 8)
board_columns = rospy.get_param("~board_columns", 11)
square_length = rospy.get_param("~square_length", 0.015)
marker_length = rospy.get_param("~marker_length", 0.011)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
board = cv2.aruco.CharucoBoard((board_columns, board_rows), square_length, marker_length, aruco_dict)
board.setLegacyPattern(True)

#Get camera intrinsic parameters
intrinsic_file = rospy.get_param("~intrinsc_file_path")
with open(intrinsic_file, "r") as f:
    data = yaml.safe_load(f)
camera_matrix = np.array(data["matrix"], dtype=np.float32)
dist_coeff = np.array(data["distorion"], dtype=np.float32)

#Transformation matrices initiliazation
T_b2ee = np.eye(4)
T_B2EE = []
rot_B2EE = []
tvect_B2EE = []
T_C2B = np.eye(4)
T_t2c = np.eye(4)
T_t2C = []
rot_t2C = []
tvect_t2C = []
pose_counter = 0
max_poses = rospy.get_param("~max_poses", 10)

#Get camera resolution parameters
#width = rospy.get_param("~camera_res_w", 640)
#height = rospy.get_param("~camera_res_h", 480)

def get_charuco_board_object_points (board, corners_ids: list | np.ndarray):
    corners = board.getChessboardCorners()
    object_points = []
    for idx in corners_ids:
        object_points.append(corners[idx])
    return np.array(object_points, dtype=np.float32)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calibration(folder_path, pose_callback):
    # Create directory to save images
    create_directory(folder_path)
     
    # Subscribe to the /terminate topic
    rospy.Subscriber('/execute_trajectory/feedback', moveit_msgs.msg.ExecuteTrajectoryActionFeedback, pose_callback, folder_path)
    rospy.spin()
    # Keep the node running


def image_streaming():
    width = rospy.get_param("~camera_res_w", 640)
    height = rospy.get_param("~camera_res_h", 480)
    # Initialize the RealSense camera pipeline
    pipeline = rs.pipeline()
    config = rs.config()
                    
    # Configure the pipeline to stream in color
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)  # Example resolution and format  try to increase resolution up to 1920, 1080
                    
    # Start streaming
    cfg =pipeline.start(config)
                               
    # Wait for a coherent color frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
                    
    if not color_frame:
        rospy.logwarn("No color frame captured.")
        return
                      
    # Convert to numpy array
    color_image = np.asanyarray(color_frame.get_data())
                    
    # Save the image
    timestamp = rospy.get_time()  # Get timestamp for unique naming
    image_name = os.path.join(folder_path, f"image_{timestamp}.png")

     # Start pipeline and get the configuration it found
    profile = cfg.get_stream(rs.stream.color) # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics()


    fx = intr.fx
    fy = intr.fy 
    cx = intr.ppx
    cy = intr.ppy 

    dist = np.array(intr.coeffs)
    matrix = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,0,1]])

    cv2.imwrite(image_name, color_image)
    rospy.loginfo(f"Image saved at {image_name}")
    
    # Stop the camera pipeline
    pipeline.stop()
    return image_name, matrix, dist


#def image_processing(image_name, frame, camera_matrix, dist_coeff, board, verbose=True):
def image_processing(image_name, frame, matrix, dist, board, verbose=True):


    #analyze the image, detecting the board. It outputs the target2camera trasnform matrix, target2camera rotation matrix, target2camera traslation vector

    #cv2.imwrite(image_name, color_image)
    #rospy.loginfo(f"Image saved at {image_name}")
    

    
    print("=> Processing image {0}".format(image_name))
    print("POSE ESTIMATION STARTS:")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    frame = cv2.imread(image_name)
    print(type(frame))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector_params = cv2.aruco.CharucoParameters()
    detector_params.minMarkers = 2
    detector_params.tryRefineMarkers = True
    detector_params.cameraMatrix = camera_matrix
    detector_params.distCoeffs = dist_coeff
    charucodetector = cv2.aruco.CharucoDetector(board, detector_params)
    charucodetector.setBoard(board)

    charuco_corners, charuco_ids, marker_corners, marker_ids = charucodetector.detectBoard(gray)       
    if marker_corners is None or marker_ids is None:
        print(f'no detection, try again')
        return None
    if len(marker_corners) != len(marker_ids) or len(marker_corners) == 0:
        print(f'wrong detection, try again')
        return None
    
    #Draw detected markers
    cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

    try:
           
        object_points = get_charuco_board_object_points(board, charuco_ids)
        charuco_corners2 = cv2.cornerSubPix(gray, charuco_corners, (23,23), (-1,-1), criteria)

        ret, p_rvec, p_tvec_t2c = cv2.solvePnP(object_points, charuco_corners2, matrix, dist)  #in camera frame (2d Image plane)

        if p_rvec is None or p_tvec_t2c is None:
            return None
        if np.isnan(p_rvec).any() or np.isnan(p_tvec_t2c).any():
            return None
        img_axes = cv2.drawFrameAxes(frame,
                            matrix, #camera_matrix,
                            dist, #dist_coeff, 
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

    new_T_t2c = np.array([[rot_t2c[0,0], rot_t2c[0,1], rot_t2c[0,2], p_tvec_t2c[0,0]],
                          [rot_t2c[1,0], rot_t2c[1,1], rot_t2c[1,2], p_tvec_t2c[1,0]],
                          [rot_t2c[2,0], rot_t2c[2,1], rot_t2c[2,2], p_tvec_t2c[2,0]],
                          [      0,        0,        0,          1]],dtype=np.float64)
    print(T_t2c)
    print(new_T_t2c)


    if verbose:
        print('Translation : {0}'.format(p_tvec_t2c))
        print('Rotation    : {0}'.format(p_rvec))
        print('Distance from camera: {0} m'.format(np.linalg.norm(p_tvec_t2c)))

    return T_t2c, rot_t2c, p_tvec_t2c,img_axes 

def pose_callback(feedback_msg, image_name, max_poses = 10):
    global pose_counter, T_B2EE, rot_B2EE, tvect_B2EE, T_t2C, rot_t2C, tvect_t2C

    if feedback_msg.feedback.state == "IDLE":  # If the /terminate signal is True
            image_name, matrix, dist = image_streaming()

            frame = cv2.imread(image_name)
            cv2.imshow('img',frame)
            cv2.waitKey(500)
            user_input = input("Do you want to save the image? (yes/no): ")

            if user_input.lower() in ["yes", "y"]:
                #print("Continuing ...") 
                rospy.loginfo("Terminate signal received. Waiting for 2 seconds before capturing image...")
                rospy.sleep(2)  # Wait for 2 seconds

                try:  
                    tfBuffer = tf2_ros.Buffer()
                    #listener = tf2_ros.TransformListener(tfBuffer)
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

                    new_T_b2ee= np.array([[rot_b2ee[0,0], rot_b2ee[0,1], rot_b2ee[0,2], tvect_b2ee[0]],
                          [rot_b2ee[1,0], rot_b2ee[1,1], rot_b2ee[1,2], tvect_b2ee[1]],
                          [rot_b2ee[2,0], rot_b2ee[2,1], rot_b2ee[2,2], tvect_b2ee[2]],
                          [      0,        0,        0,          1]],dtype=np.float64)

                    print(T_b2ee)
                    print(f'b2ee')
                    print(new_T_b2ee)     
                    
                except Exception as e:
                    print(f"Transform Error: {e}")
                                    
                #Capture and process the image
                if image_name:
                    frame = cv2.imread(image_name)
                    T_t2c, rot_t2c, tvect_t2c, img_axes  = image_processing(image_name,frame, matrix, dist, board, verbose =True)
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
                        rot_t2C_mat = np.array(rot_t2C) 
                        tvect_t2C_mat = np.array(tvect_t2C)
                        print(f"rot_t2C_mat shape: {rot_t2C_mat.shape}")
                        print(f"tvect_t2C_mat shape: {tvect_t2C_mat.shape}") 

                        #store the matrix in an array
                        T_B2EE.append(T_b2ee)
                        rot_B2EE.append(rot_b2ee) 
                        tvect_B2EE.append(tvect_b2ee)  
                        
                        #convert from list to array
                        rot_B2EE_mat = np.array(rot_B2EE) 
                        tvect_B2EE_mat = np.array(tvect_B2EE)
                    pose_counter += 1
                    rospy.loginfo(f"Pose {pose_counter} acquired successfully")

                    if pose_counter >= max_poses:   
                        rot_c2b, t_vect_c2b = cv2.calibrateHandEye(rot_B2EE_mat,tvect_B2EE_mat,rot_t2C_mat, tvect_t2C_mat, cv2.CALIB_HAND_EYE_PARK)

                        #assembly the transformation matrix
                        T_C2B[:3, :3] = rot_c2b
                        T_C2B[:3, 3] = t_vect_c2b.T   
                        
                        #rot_c2b_vect = cv2.Rodrigues(rot_c2b)

                        new_T_C2B = np.array([[rot_c2b[0,0], rot_c2b[0,1], rot_c2b[0,2], t_vect_c2b[0,0]],
                          [rot_c2b[1,0], rot_c2b[1,1], rot_c2b[1,2], t_vect_c2b[1,0]],
                          [rot_c2b[2,0], rot_c2b[2,1], rot_c2b[2,2], t_vect_c2b[2,0]],
                          [      0,        0,        0,          1]],dtype=np.float64)
                        

                        print("Final Transformation Matrix (Camera to robot's base):")
                        print(T_C2B)

                        return T_C2B
            else:
                print("Change robot's pose and try again")
                return None

#Save results in a yaml file 
saving_directory = './calibration_results/'
create_directory(saving_directory)
output_file = os.path.join(saving_directory, "extrinsic_rs.yaml")

data = {
    'Camera' : "Realsense", 
    'Extrinsic matrix': T_C2B,
    'Poses acquired' : max_poses,
}

with open(output_file, "w") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print(f'Results saved in ', output_file)

if __name__ == '__main__':
    rospy.init_node('calibrator', anonymous=False)

    try:
        folder_path = './extrinsic_rs_images'
        calibration(folder_path, pose_callback)

    except rospy.ROSInterruptException:
        pass

