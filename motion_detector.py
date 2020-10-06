import cv2
from datetime import datetime
import pandas as pd

time = []
first_frame = None
status_list = [None,None]
video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
df = pd.DataFrame(columns=["Start","End"])

while True:
    check,frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # converted into gray frame
    gray = cv2.GaussianBlur(gray,(21,21),0) #makes gray image blury

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray) #looks for difference between frames

    thresh_frame = cv2.threshold(delta_frame,30,255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame,None,iterations=2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #external contours

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y), (w+x, y+h), (0,255,0),3)



    status_list.append(status)
    if status_list[-1] == 1 and status_list[-2] == 0:
        time.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        time.append(datetime.now())
    

    cv2.imshow("Capturing", gray)
    cv2.imshow("delta",delta_frame)
    cv2.imshow("Threshhold Frame",thresh_frame)
    cv2.imshow("Color Frame",frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        if status == 1:
            time.append(datetime.now())
        break


for i in range(0,len(time),2):
    df = df.append({"Start": time[i], "End": time[i+1]}, ignore_index=True)

df.to_csv("Time.csv")
video.release() 
cv2.destroyAllWindows()
