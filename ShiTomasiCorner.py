import numpy as np
import cv2

# Read image in as grayscale
img = cv2.imread('Corner1.jpg')
#img = cv2.imread('Corner2.jpg')
#img = cv2.imread('Corner3.jpg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])

# Test 1
corners = cv2.goodFeaturesToTrack(gray,100,0.01,10)
corners = np.int0(corners)
out = img
for i in corners:
    x,y = i.ravel()
    cv2.circle(out,(x,y),3,255,-1)
cv2.imwrite('1_100_0.01_10.jpg',out)

# Test 2
corners = cv2.goodFeaturesToTrack(gray,100,0.1,10)
corners = np.int0(corners)
out = img
for i in corners:
    x,y = i.ravel()
    cv2.circle(out,(x,y),3,255,-1)
cv2.imwrite('1_100_0.1_10.jpg',out)

# Test 3
corners = cv2.goodFeaturesToTrack(gray,100,0.01,5)
corners = np.int0(corners)
out = img
for i in corners:
    x,y = i.ravel()
    cv2.circle(out,(x,y),3,255,-1)
cv2.imwrite('1_100_0.01_5.jpg',out)