import cv2
import numpy as np

cap = cv2.VideoCapture('../v_Bowling_g21_c03.avi')
#cap = cv2.VideoCapture('../v_GolfSwing_g18_c05.avi')
#cap = cv2.VideoCapture('../v_CricketShot_g04_c01.avi')

rgb = []
flow = []

ret, frame1 = cap.read()

# # Normalised [-1,1]
# frame1 = 2*(frame1 - np.min(frame1))/np.ptp(frame1)-1

prvs = cv2.resize(frame1, (224,224))
prvs = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)

while(1):
    ret, frame2 = cap.read()
    if ret==True:
        # # Normalised [-1,1]
        # frame2 = 2*(frame2 - np.min(frame2))/np.ptp(frame2)-1

        next = cv2.resize(frame2, (224,224))
        rgb.append(next)

        next = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)

        frame_flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow.append(frame_flow)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = next
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

