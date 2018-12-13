import numpy as np
import cv2

# Read image in as grayscale
#img = cv2.imread('Corner1.jpg')
#img = cv2.imread('Corner2.jpg')
img = cv2.imread('Corner3.jpg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# height = np.size(img,0)
# width = np.size(img,1)

# cv2.cornerHarris(img, blockSize, ksize, k)

# Test 1
dst = cv2.cornerHarris(gray,2,3,0.04)
out = img
out[dst>0.01*dst.max()]=[0,0,255]
cv2.imwrite('3_2_3_0.04.jpg',out)

# Test 2
dst = cv2.cornerHarris(gray,5,3,0.04)
out = img
out[dst>0.01*dst.max()]=[0,0,255]
cv2.imwrite('3_5_3_0.04.jpg',out)

# Test 3
dst = cv2.cornerHarris(gray,2,7,0.04)
out = img
out[dst>0.01*dst.max()]=[0,0,255]
cv2.imwrite('3_2_7_0.04.jpg',out)