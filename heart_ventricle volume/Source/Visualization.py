import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
import matplotlib

from OpenGL.GL import *
from OpenGL.GLU import *


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

img11 = cv2.imread('fig1.jpg',1)
img1 =cv2.resize(img11,(200,200))
img22 = cv2.imread('fig2.jpg',1)
img2 = cv2.resize(img22,(200,200))

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]


stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
#disparity = stereo.compute(img1,img2)
#plt.imshow(disparity,'gray')
# plt.show()
frame1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
frame2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print('computing disparity...')
disp = stereo.compute(frame1, frame2).astype(np.float32) / 16.0

h, w = img2.shape[:2]
f = 1.0*w                          # guess for focal length
Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])

points = cv2.reprojectImageTo3D(disp, Q)
colors = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
mask = disp > disp.min()
out_points = points[mask]
out_colors = colors[mask]

pygame.init()
display = (1100, 800)
screen=pygame.display.set_mode(display,DOUBLEBUF|OPENGL)

gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

glTranslatef(0.0, 0.0, -5)



while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glBegin(GL_POINTS)
    for vertex in range(0,len(out_points)):
        glVertex3fv(out_points[vertex])
        glColor3fv(out_colors[vertex])
    glEnd()

    pygame.display.flip()
    pygame.time.wait(10)

