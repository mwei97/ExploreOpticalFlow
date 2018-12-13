import cv2
import numpy as np
from flowlib import flow_to_image
import datetime

print(datetime.datetime.now())

cap = cv2.VideoCapture('../v_Bowling_g21_c03.avi')
#cap = cv2.VideoCapture('../v_GolfSwing_g18_c05.avi')
#cap = cv2.VideoCapture('../v_CricketShot_g04_c01.avi')

optical_flow = cv2.DualTVL1OpticalFlow_create()

rgb = []
flow = []

ret, frame = cap.read()

# # Normalised [-1,1]
# frame = 2*(frame - np.min(frame))/np.ptp(frame)-1

prvs_img = cv2.resize(frame, (224,224)) # shape(224,224,3)
prvs_img = cv2.cvtColor(prvs_img, cv2.COLOR_BGR2GRAY) # shape(224,224)

while(1):
	ret, frame = cap.read()
	if ret==True:	
		# # Normalised [-1,1]
		# frame = 2*(frame - np.min(frame))/np.ptp(frame)-1
		
		next_img = cv2.resize(frame,(224,224))
		rgb.append(next_img)
		next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

		frame_flow = optical_flow.calc(prvs_img, next_img, None) # shape(224,224,2)
		flow.append(frame_flow)

		k = cv2.waitKey(30) & 0xff
		if k==27:
			break
		prvs_img = next_img
	else:
		break

cap.release()
cv2.destroyAllWindows()

tmp = []
tmp.append(rgb)
rgb = np.array(tmp)

tmp = []
tmp.append(flow)
flow = np.array(tmp)

np.save('rgb', rgb)
np.save('flow', flow)

print(datetime.datetime.now())

