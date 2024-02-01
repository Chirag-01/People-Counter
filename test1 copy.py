import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import pandas as pd
import numpy as np
import cvzone
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture("Untitled video - Made with Clipchamp.mp4")
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
count=0
going_out={}
going_in={}
cnt1=[]
cnt2=[]

area1=[(593,344),(642,367),(701,230),(652,228)]   #484
area2=[(644, 380),(684, 393),(749,246),(709,232)]   #484



while cap.isOpened():
    success, frame = cap.read()
    frame=cv2.resize(frame,(1020,500))
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True,conf=0.5,tracker="bytetrack.yaml")
        a=results[0].boxes.data
        px=pd.DataFrame(a).astype("float")
        # print(px)
        list=[]
        for index,row in px.iterrows():
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])            
            c=class_list[d]
            if 'person' in c:
                if results[0].boxes.id is not None:
                    list.append([x1,y1,x2,y2,int(results[0].boxes.id[0])])
                    xc=int(x1+x2)//2
                    yc=int(y1+y2)//2
                    result=cv2.pointPolygonTest(np.array(area2,np.int32),((xc,yc)),False)
                    print("Result is",results[0].boxes.id)
                    for j in results[0].boxes.id:
                        id=int(j)
                        print("Result is",result,id)
                        if result >=0:
                            going_out[id]=(xc,yc)   
                        print(going_out)
                        if id in going_out:
                            result1=cv2.pointPolygonTest(np.array(area1,np.int32),((xc,yc)),False)
                            if result1>=0:
                                cv2.circle(frame,(xc,yc),7,(255,0,255),-1)         
                                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
                                if cnt1.count(id)==0:
                                    cnt1.append(id)
                                
        # Visualize the results on the frame
        # print(annotated_frame)
        print('cnt1:',len(cnt1))
        print('cnt2:',len(cnt2))
        out_c=len(cnt1)
        in_c=len(cnt2)
    
        cvzone.putTextRect(frame,f'exit:{out_c}',(50,60),2,2)
        cvzone.putTextRect(frame,f'entry:{in_c}',(50,160),2,2)
        cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
        cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)

        cv2.imshow("YOLOv8 Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()