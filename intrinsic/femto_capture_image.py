#!/usr/bin/env python
import cv2
import pyorbbecsdk as femto 
from pyorbbecsdk import OBSensorType, OBFormat, FrameSet, OBError, VideoStreamProfile
#from utils import frame_to_bgr_image
import numpy as np
import time
import os

#This script acquires 30 images with of the chessboard for intrinsic calibration of a femtoBolt camera
#NB import cv2 before pyorbbecsdk

#Create directory for saving results
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#Set resolution and capture images
def capture_images(folder_path, interval=5, max_counter=30, width = 1280, height = 720):
    create_directory(folder_path)
    
    counter = 0
    pipeline = femto.Pipeline()    
    config = femto.Config()

    #Start stream pipeline for available color profile
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(width, height, OBFormat.RGB, 30)
        except OBError as e:
            #print(e)
            #Switch to default profile if not available
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
    pipeline.start(config)

    #Acquire frames
    try:
        while True: 
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()

            #Convert to array
            col_image = np.asanyarray(color_frame.get_data())
            reshaped_image = col_image.reshape((height, width, 3))
            color_image= cv2.cvtColor(reshaped_image, cv2.COLOR_BGR2RGB)
    
            #Save image with parametric name
            current_time = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(folder_path, f'image_{current_time}.png')
            cv2.imwrite(filename, color_image)
            counter = counter + 1 
            print(counter, f' image saved')

            time.sleep(interval)
            if counter == max_counter:
                break

    except KeyboardInterrupt:
        print("Acquisition stopped by the user")

if __name__== "__main__":
    rel_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(rel_path,"../images_femto")
    capture_images(folder_path, interval=5, max_counter=30)