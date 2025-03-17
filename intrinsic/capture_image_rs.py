#!/usr/bin/env python
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

#This script acquires 30 images with of the chessboard for intrinsic calibration of a realsense camera

#Create directory for saving results
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#Set resolution and capture images
def capture_images(folder_path, interval=5, max_counter=30, width = 848, height = 480):
    create_directory(folder_path)
    counter = 0
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    #Get default intrinsic parameters
    cfg = pipeline.start(config)
    profile = cfg.get_stream(rs.stream.color)
    intr = profile.as_video_stream_profile().get_intrinsics()
    
    fx = intr.fx
    fy = intr.fy 
    cx = intr.ppx
    cy = intr.ppy 
    #dist_coeff = np.array(intr.coeffs)
    #matrix = np.array([[fx, 0, cx],
                       #[0, fy, cy],
                       #[0,  0,  1]])
    
    try:
        while True: 
            frames= pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            #Convert to array
            color_image = np.asanyarray(color_frame.get_data())

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
    folder_path = os.path.join(rel_path,"../images_rs")
    capture_images(folder_path, interval=5, max_counter=30)

