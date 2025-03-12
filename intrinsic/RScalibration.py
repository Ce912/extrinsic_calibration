import numpy as np
import cv2 as cv
import glob
import yaml
import os

#This script perfmor intrisic calibration of a realsense camera exploiting the images saved in './images_rs/' 
#and store results in a .yaml file.
#More info on the OpenCv page
#CAVEAT: rename one of the image as "distorted.png" to see the results

#Set grid parameters
grid_columns = 8
grid_rows = 6
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
objp = np.zeros((grid_rows*grid_columns,3), np.float32)
objp[:,:2] = np.mgrid[0:grid_columns,0:grid_rows].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('./images_rs/*.png')
 
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (grid_columns,grid_rows), None)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cv.drawChessboardCorners(img, (grid_columns,grid_rows), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#Evaluate a distorted pn
img = cv.imread('./images_rs/distorted.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

new_h, new_w = dst.shape[:2]

# Padding dimensions (we'll pad to the original size)
top_padding = (h - new_h) // 2
bottom_padding = h - new_h - top_padding
left_padding = (w - new_w) // 2
right_padding = w - new_w - left_padding

# Pad the image with black borders (you can change the color if needed)
dst_padded = cv.copyMakeBorder(dst, top_padding, bottom_padding, left_padding,right_padding, cv.BORDER_CONSTANT, value=(0, 0, 0))
 
#Save undistorted image
cv.imwrite('./images_rs/corrected.png', dst_padded)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

total_error = mean_error/len(objpoints)

print( "total error: {}".format(mean_error/len(objpoints)) )

#Set directory for saving results
directory = "./calibration_results"
if not os.path.exists(directory):
    os.makedirs(directory)

output_file = os.path.join(directory, "intrinsic_rs.yaml")
mtx = mtx.tolist()
dist = dist.tolist()
data = {
    "Camera": "Realsense",
    "totatl error ": total_error,
    "ret": ret,
    "matrix": mtx, 
    "distorion" : dist,
}
print(data)

with open(output_file, "w") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print(f'Results saved in ', output_file)

cv.destroyAllWindows()

