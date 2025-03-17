import numpy as np
import cv2 as cv
import glob
import yaml
import os

#This script perfmor intrisic calibration of a realsense camera exploiting the images saved in './images_femto/' 
#and store results in a .yaml file.
#More info on the OpenCv page
#CAVEAT: rename one of the image as "distorted.png" to see the results

#Set grid parameters
grid_columns = 8
grid_rows = 6
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((grid_rows*grid_columns,3), np.float32)
objp[:,:2] = np.mgrid[0:grid_columns,0:grid_rows].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
#Define path to images
rel_path = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(rel_path,"../images_femto/*.png")
images = glob.glob(image_path)
 
#Read images in the folder
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

#Estimate camera matrix and dist coeff
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#Evaluate a distorted image
dist_path = os.path.join(rel_path,"../images_femto/distorted.png")
img = cv.imread(dist_path)
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
corr_path = os.path.join(rel_path,"../images_femto/corrected.png")
cv.imwrite(corr_path, dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

#Set directory for saving results
rel_ = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(rel_, "./calibration_results")
if not os.path.exists(directory):
    os.makedirs(directory)

output_file = os.path.join(directory, "intrinsic_femto.yaml")
total_error = mean_error/len(objpoints)

#Convert to list for saving in yaml
mtx = mtx.tolist()
dist = dist.tolist()
data = {
    "Camera": "FemtoBolt",
    "totatl error ": total_error,
    "ret": ret,
    "matrix": mtx, 
    "distorion" : dist,
    "rvecs" : rvecs,
    "tvects" : tvecs,
}

with open(output_file, "w") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
print(f'Results saved in ', output_file)
 
cv.destroyAllWindows()


