import cv2
import numpy as np
from flowlib import flow_to_image

#cap = cv2.VideoCapture('../v_Bowling_g21_c03.avi')
#cap = cv2.VideoCapture('../v_GolfSwing_g18_c05.avi')
#cap = cv2.VideoCapture('../v_CricketShot_g04_c01.avi')
#cap = cv2.VideoCapture('../cook1.avi')
cap = cv2.VideoCapture('../cook2.avi')

optical_flow = cv2.DualTVL1OpticalFlow_create()

ret, frame = cap.read()
prvs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame)
hsv[...,1] = 255

while(1):
	ret, frame = cap.read()
	if ret==True:	
		next_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		flow = optical_flow.calc(prvs_img, next_img, None)

		rgb = flow_to_image(flow)

		# mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		# hsv[...,0] = ang*180/np.pi/2
		# hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		# rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

		if int(cap.get(1))%5==0:
			img_name = 'cook2'+str(int(cap.get(1)))+'.jpg'
			cv2.imwrite(img_name, rgb)

		k = cv2.waitKey(30) & 0xff
		if k==27:
			break
		prvs_img = next_img
	else:
		break

cap.release()
