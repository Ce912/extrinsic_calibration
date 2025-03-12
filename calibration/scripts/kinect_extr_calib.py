#!/usr/bin/env python
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2, os
from cv2 import aruco
import rospy 
import tf2_ros
import glob
#import pyrealsense2 as rs
import freenect2
from freenect2 import Device, FrameType, Frame
import geometry_msgs as geom_msg
from std_msgs.msg import Bool
import moveit_msgs.msg 
import tf



images = glob.glob('/home/leon/shared_ws/cecilia_ws/images_kinect/*.png')
#board = glob.glob('/home/ce912/Calibration/*.pdf')
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
board = cv2.aruco.CharucoBoard((11, 8), 0.015, 0.011, aruco_dict)
board.setLegacyPattern(True)
Nposes = 20




def get_charuco_board_object_points (board, corners_ids: list | np.ndarray):
    corners = board.getChessboardCorners()
    object_points = []
    for idx in corners_ids:
        object_points.append(corners[idx])
    return np.array(object_points, dtype=np.float32)




# kinect camera color parameters 

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



#Nposes = len(images)

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


def image_streaming():
    device = Device()
    #device.open()
    
    device.color_camera_params = freenect2.ColorCameraParams() #camera color intrinsic calibration parameters
    device.color_camera_params.fx = color_fx
    device.color_camera_params.fy = color_fy
    device.color_camera_params.cx = color_cx
    device.color_camera_params.cy = color_cy

    device.ir_camera_params = freenect2.IrCameraParams() #camera IR intrinsic calibration parameters
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
        color, depth, ir = get_next_frames(device)
        #listener = device.color_frame_listener(FrameType.Color, Frame.bytes_per_pixel)
        
        #rospy.loginfo('listener')
        #device.start(frame_listener=listener)
        #rospy.loginfo('list')
        
        #frames_obj = device.get_next_frame(timeout=2)
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
            device.close()

                
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
            


            
                            
            # Save the image
            timestamp = rospy.get_time()  # Get timestamp for unique naming
            image_name = os.path.join(folder_path, f"image_{timestamp}.png")
            rospy.loginfo(f"prova")

            cv2.imwrite(image_name, color_frame)
            rospy.loginfo(f"Image saved at {image_name}")
            
            
            # Stop the camera pipeline
    return image_name


def image_processing(image_name, frame, color_camera_matrix, color_dist_coeff, board, verbose=True):

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
    detector_params.cameraMatrix = color_camera_matrix
    detector_params.distCoeffs = color_dist_coeff
    charucodetector = cv2.aruco.CharucoDetector(board, detector_params)
    charucodetector.setBoard(board)

    charuco_corners, charuco_ids, marker_corners, marker_ids = charucodetector.detectBoard(gray)


    #detectorParams = cv2.aruco.DetectorParameters()
    #detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
    #corners, ids, rejectedImgPoints = cv2.aruco.ArucoDetector.detectMarkers(gray, dictionary)
    #marker_corners, marker_ids, rejected = detector.detectMarkers(frame)


    #cv2.imshow('img', frame)
    #cv2.waitKey(500)
       
    if marker_corners is None or marker_ids is None:
        print(f'no detection')
        return None
    if len(marker_corners) != len(marker_ids) or len(marker_corners) == 0:
        print(f'no detection2')
        return None
    
    #Draw detected markers
    cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

    try:
            #charucodeimage_name = tector = cv2.aruco.CharucoDetector(board)
            #det_corners, det_ids, _, _ = charucodetector.detectBoard(frame)
                
            #p_rvec [3x1], p_tvec [3x1]
            #det_corners_mat = cv2.Mat(det_corners)
        #objPoints, imgPoints = cv2.aruco.CharucoBoard.matchImagePoints(allCorners,allIds)

        object_points = get_charuco_board_object_points(board, charuco_ids)
        charuco_corners2 = cv2.cornerSubPix(gray, charuco_corners, (23,23), (-1,-1), criteria)
        ret, p_rvec, p_tvec_t2c = cv2.solvePnP(object_points, charuco_corners2, color_camera_matrix,color_dist_coeff)  #in camera frame (2d Image plane)

        


        if p_rvec is None or p_tvec_t2c is None:
            return None
        if np.isnan(p_rvec).any() or np.isnan(p_tvec_t2c).any():
            return None
        img_axes = cv2.drawFrameAxes(frame,
                            color_camera_matrix,
                            color_dist_coeff,
                            p_rvec,
                            p_tvec_t2c,
                            0.1)
            #cv2.aruco.drawDetectedCornersCharuco(frame, c_corners, c_ids)
            #cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            #cv2.aruco.drawDetectedMarkers(frame, rejected_points, borderColor=(100, 0, 240))
        #cv2.imshow('img with axes', img_axes)
        #input('printed image with axes')
        

    except cv2.error:
        return None
        
    #rotation = R.from_rotvec(p_rvec.T)

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
    #input('T_t2c')
    print(new_T_t2c)
    #input('compare T_t2c and new_T_t2c, they should be the same') #[checked]

    if verbose:
        print('Translation : {0}'.format(p_tvec_t2c))
        print('Rotation    : {0}'.format(p_rvec))
        print('Distance from camera: {0} m'.format(np.linalg.norm(p_tvec_t2c)))

    return T_t2c, rot_t2c, p_tvec_t2c,img_axes  #p_tvec_t2c is a 3x1 but it's a multidimensional array



def pose_callback(feedback_msg, image_name):
    global pose_counter, T_B2EE, rot_B2EE, tvect_B2EE, T_t2C, rot_t2C, tvect_t2C

    if feedback_msg.feedback.state == "IDLE":  # If the /terminate signal is True
            image_name = image_streaming()

            frame = cv2.imread(image_name)
            print(frame.shape)
            rospy.loginfo('check image')
            cv2.imshow('img',frame)
            cv2.waitKey(500)

            user_input = input("Do you want to save the image? (yes/no): ")

            if user_input.lower() in ["yes", "y"]:
                print("Continuing ...")

                
                rospy.loginfo("Terminate signal received. Waiting for 2 seconds before capturing image...")
                rospy.sleep(2)  # Wait for 2 seconds

                try: 
                    
                    #listener.waitForTransform("panda_EE", "panda_link0", rospy.Time(), rospy.Duration(4.0))
                    tfBuffer = tf2_ros.Buffer()
                    listener = tf2_ros.TransformListener(tfBuffer)
                
                    rospy.sleep(4)
            
                    #Get the transform from base reference frame to gripper reference frame
                    #res2 = listener.lookupTransform("panda_EE", "panda_link0",rospy.Time(0))
                    res = tfBuffer.lookup_transform("panda_EE", "panda_link0",rospy.Time(0))

                    #input('prova compilazione')
                    print(type(res))
                    print(res.transform)
                    
                    
                    #tvect_ee2b = np.array([res.transform.translation.x,res.transform.translation.y, res.transform.translation.z])
                    #rot_ee2b = np.array([res.transform.rotation.x,res.transform.rotation.y, res.transform.rotation.z, res.transform.rotation.w])
                    #r= R.from_quat(rot_ee2b)
                    #rot_ = r.as_matrix()

                    #T_b2ee[:3, :3] = rot_.T
                    #T_b2ee[:3, 3] = -np.dot(rot_.T, tvect_ee2b.T)
                    
                    
                    tvect_b2ee = np.array([res.transform.translation.x,res.transform.translation.y, res.transform.translation.z]) 
                    rot = np.array([res.transform.rotation.x,res.transform.rotation.y, res.transform.rotation.z, res.transform.rotation.w])


                    r = R.from_quat(rot)
                    rot_b2ee = r.as_matrix()
                    

                    #assembly the transformation matrix
                    T_b2ee[:3, :3] = rot_b2ee
                    T_b2ee[:3, 3] = tvect_b2ee.T   #check dimension

                    new_T_b2ee= np.array([[rot_b2ee[0,0], rot_b2ee[0,1], rot_b2ee[0,2], tvect_b2ee[0]],
                          [rot_b2ee[1,0], rot_b2ee[1,1], rot_b2ee[1,2], tvect_b2ee[1]],
                          [rot_b2ee[2,0], rot_b2ee[2,1], rot_b2ee[2,2], tvect_b2ee[2]],
                          [      0,        0,        0,          1]],dtype=np.float64)

                    print(T_b2ee)
                    print(f'b2ee')
                    print(new_T_b2ee)
                    #input('compare b2ee and new t_b2ee, they should be the same') #[checked]
        
                    
                    
                except Exception as e:
                    print(f"Transform Error: {e}")
                    

                
                #Capture and process the image

                #image_name = image_streaming()  #old version: uncommented

                if image_name:
                    frame = cv2.imread(image_name)
                    T_t2c, rot_t2c, tvect_t2c, img_axes = image_processing(image_name,frame, color_camera_matrix, color_dist_coeff, board, verbose =True)
                    cv2.imshow('img with axes',img_axes)
                    cv2.waitKey(500)

                    #rot_t2c is a multidimensional rotation matrix (3x3,), tvect_t2c is a cmultidimensional column vector (3,) 
                    # T_t2c is a 4x4 trasformation matrix

                    if T_t2c is not None:
                        tvect_t2c = tvect_t2c.flatten() 
                        T_t2C.append(T_t2c)
                        rot_t2C.append(rot_t2c) #3x3
                        tvect_t2C.append(tvect_t2c) #old version = tvect_t2c.T

                        #print(rot_t2C)
                        #print(tvect_t2C)
                        
                        #convert from list to array
                        rot_t2C_mat = np.array(rot_t2C) #column array of 3x3 rotation matrices--> check and eventually transpose to have matrices next to each other
                        
                        tvect_t2C_mat = np.array(tvect_t2C) # matrix of column translation vectors stack one next to each other
                        print(rot_t2C_mat)
                        print(f"rot_t2C_mat shape: {rot_t2C_mat.shape}") #expected: 3d array of (nposes,3,3)
                        print(tvect_t2C_mat)
                        print(f"tvect_t2C_mat shape: {tvect_t2C_mat.shape}")   #expected: 2d array (nposes,3)  #check, it's nposes,3,1
                        #input('check dimension')

                        #store the matrix in an array
                        T_B2EE.append(T_b2ee)
                        rot_B2EE.append(rot_b2ee)  #check!!! old version = rot_b2ee.T
                        tvect_B2EE.append(tvect_b2ee) # 
                        
                        #convert from list to array
                        rot_B2EE_mat = np.array(rot_B2EE) #column array of 3x3 rotation matrices--> check and eventually transpose to have matrices next to each other
                        #tvect_B2EE_mat = np.array(tvect_B2EE).T #matrix of column translation vectors stack one next to each other--> old version 3xnposes
                        tvect_B2EE_mat = np.array(tvect_B2EE)
                        print(tvect_B2EE_mat)
                        print(f"tvect_B2EE_mat shape: {tvect_B2EE_mat.shape}")  #expected: 2d array (nposes,3) 
                        #input('traslation vector matrix')
                        print(rot_B2EE_mat)
                        print(f"rot_B2EE_mat shape: {rot_B2EE_mat.shape}")    #expected: 3d array of (nposes,3,3)
                        #input('array of rotation matrices, stack one below each other')
                    

                    pose_counter += 1
                    rospy.loginfo("Pose {pose_counter} acquired successfully")
                    if pose_counter >= Nposes:   #avoid 3 to better check on the dimensions
                        #compute rotation matrix and traslation vector
                        print(rot_B2EE_mat)
                        print(tvect_B2EE_mat)
                        input("trasformate lato robot")

                        print(rot_t2C_mat)
                        print(tvect_t2C_mat)
                        input("trasformate lato camera")

                        rot_c2b, t_vect_c2b = cv2.calibrateHandEye(rot_B2EE_mat,tvect_B2EE_mat,rot_t2C_mat, tvect_t2C_mat, cv2.CALIB_HAND_EYE_PARK)
                        
                        #rot_c2b is a 3x3 rotation matrix
                        #t_vect_c2b is a 3x1 traslation vector
                    
                        #assembly the transformation matrix
                        T_C2B[:3, :3] = rot_c2b
                        T_C2B[:3, 3] = t_vect_c2b.T    #why transpose? CHECK

                        #rot_ee2c_ = R.from_matrix(rot_ee2c)
                        #rot_ee2c_vect =R.as_rotvec(rot_ee2c_)
                        
                        rot_c2b_vect = cv2.Rodrigues(rot_c2b)

                        new_T_C2B = np.array([[rot_c2b[0,0], rot_c2b[0,1], rot_c2b[0,2], t_vect_c2b[0,0]],
                          [rot_c2b[1,0], rot_c2b[1,1], rot_c2b[1,2], t_vect_c2b[1,0]],
                          [rot_c2b[2,0], rot_c2b[2,1], rot_c2b[2,2], t_vect_c2b[2,0]],
                          [      0,        0,        0,          1]],dtype=np.float64)
                        
                        print(T_C2B)
                        input('T_C2B')
                        print(new_T_C2B)
                        input('Check if the two matrices are the same')

                        print("Final Transformation Matrix (End-Effector to Camera):")
                        print(T_C2B)
                        print(t_vect_c2b)
                        print(f"tvect_c2b shape: {t_vect_c2b.shape}")
                        input('check dimension of the vector: column or multidimensional?')
            else:
                print("Change robot's pose and try again")




# Usage example

if __name__ == '__main__':
    rospy.init_node('calibrator', anonymous=False)  #anonymous = True before
    #rospy.spin()
    try:
        folder_path = '/home/leon/shared_ws/cecilia_ws/images'
        calibration(folder_path, pose_callback)

    except rospy.ROSInterruptException:
        pass

#Transformation from frame-to-frame to get camera to robot's base transform
