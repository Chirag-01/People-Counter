import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import torch
import json
import threading
import time
import psycopg2 as spg
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime
import json
import torch

DB_HOST = "localhost"
DB_USER = "postgres"
DB_PASS = "root"

def connect_to_db(DB_NAME="new db"):
    # Connect to the DATABASE
    con = spg.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port='5432'
    )
    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    # Cursor
    cur = con.cursor()
    cur.execute("select version()")
    data = cur.fetchone()
    print("Connection established to:", data)
    print(con)
    print(cur)
    return con, cur

# Check if GPU is available and set device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load YOLO model
model = YOLO('yolov8s.pt')

# Move the model to the GPU if available
model.to(device)



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  

def track_person(stream_data, cam_id,ip,area):
    print(cam_id,"iod")
    stream_link=stream_data['streamLink']
    cam_coord=stream_data['camera']
    cx1,cy1,cx2,cy2=cam_coord
    # db_connection, db_cursor = connect_to_db()

    # cx1,cy1,cx2,cy2=cam_coord[0],cam_coord[1],cam_coord[2],cam_coord[3]
    cv2.namedWindow(f'RGB{cam_id}')
    cv2.setMouseCallback(f'RGB{cam_id}', RGB)
    cap=cv2.VideoCapture(stream_link)


    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n") 
    #print(class_list)

    count=0
    persondown={}
    tracker=Tracker()
    counter1=[]

    personup={}
    counter2=[]

    offset=10
    while True:    
        ret,frame = cap.read()
        if not ret:
            break
    #    frame = stream.read()

        count += 1
        if count % 3 == 0:
            continue
        frame=cv2.resize(frame,(1020,500))
    

        results=model.track(frame)
    #   print(results)
        a=results[0].boxes.data
        px = pd.DataFrame(a.cpu().numpy()).astype("float")

    #    print(px)
        list=[]
    
        for index,row in px.iterrows():
    #        print(row)
    
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            
            c=class_list[d]
            if 'person' in c:

                list.append([x1,y1,x2,y2])
        
            
        bbox_id=tracker.update(list)
        for bbox in bbox_id:
            x3,y3,x4,y4,id=bbox
            cx=int(x3+x4)//2
            cy=int(y3+y4)//2
            if cam_id=="484": # Vertical
                                # for Down ⬇
                # print("downb ex")
                # print(cx1-offset,cx<cx1,offset,"1")
                if cx1<(cx+offset) and cx1>(cx-offset):
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                    cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                    persondown[id]=(cx,cy)

                # print(cx2-offset,cx,cx2-offset,"2")
                if id in persondown:
                    
                    if cx2+5<(cx+offset) and cx2>(cx-offset):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
                        cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                        if counter1.count(id)==0:
                            counter1.append(id)

                # for up ⬆
                # print("up")
                
                if cx2<(cx+offset) and cx2>(cx-offset):
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,0),2)
                    cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                    personup[id]=(cx,cy)
                    # print("id",id)
                # print(personup)
               
                if id in personup:
                    # print(cx,cx1+offset,"hsjs")
                    if cx1<(cx+offset) and cx1>(cx-offset):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
                        cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                        if counter2.count(id)==0:
                            counter2.append(id)
                            

                cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
            elif cam_id=="481": # Horizontal
                #for Down ⬇
                offset=10
                if cy1<(cy+offset) and cy1>(cy-offset):
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                    cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                    persondown[id]=(cx,cy)
                if id in persondown:
                    if cy2<(cy+offset) and cy2>(cy-offset):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
                        cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                        if counter1.count(id)==0:
                            counter1.append(id)
                #for up ⬆
                if cy2<(cy+offset) and cy2>(cy-offset):
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                    cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                    personup[id]=(cx,cy)

                if id in personup:
                    if cy1<(cy+offset) and cy1>(cy-offset):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
                        cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                        if counter2.count(id)==0:
                            counter2.append(id)
                            

                cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
            elif cam_id=="482": # Horizontal
                               # for Down ⬇
                # print("downb ex")
                # print(cx1-offset,cx<cx1,offset,"1")
                if cx1<(cx+offset) and cx1>(cx-offset):
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                    cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                    persondown[id]=(cx,cy)

                # print(cx2-offset,cx,cx2-offset,"2")
                if id in persondown:
                    
                    if cx2<(cx+offset) and cx2>(cx-offset):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
                        cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                        if counter1.count(id)==0:
                            counter1.append(id)

                # for up ⬆
                # print("up")
                
                if cx2<(cx+offset) and cx2>(cx-offset):
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,0),2)
                    cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                    personup[id]=(cx,cy)
                    # print("id",id)
                # print(personup)
               
                if id in personup:
                    # print(cx,cx1+offset,"hsjs")
                    if cx1<(cx+offset) and cx1>(cx-offset):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
                        cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                        if counter2.count(id)==0:
                            counter2.append(id)
                            

                cv2.circle(frame,(cx,cy),4,(255,0,255),-1)

            elif cam_id=="487": #Vertical
                # for Down ⬇
                # print("downb ex")
                # print(cx1-offset,cx<cx1,offset,"1")
                if cx1<(cx+offset) and cx1>(cx-offset):
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                    cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                    persondown[id]=(cx,cy)

                # print(cx2-offset,cx,cx2-offset,"2")
                if id in persondown:
                    
                    if cx2<(cx+offset) and cx2>(cx-offset):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
                        cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                        if counter1.count(id)==0:
                            counter1.append(id)

                # for up ⬆
                # print("up")
                
                if cx2<(cx+offset) and cx2>(cx-offset):
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,0),2)
                    cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                    personup[id]=(cx,cy)
                    # print("id",id)
                # print(personup)
               
                if id in personup:
                    # print(cx,cx1+offset,"hsjs")
                    if cx1<(cx+offset) and cx1>(cx-offset):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
                        cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                        if counter2.count(id)==0:
                            counter2.append(id)
                            

                cv2.circle(frame,(cx,cy),4,(255,0,255),-1)

            
        if cam_id=="484": # Vertical
            cv2.line(frame,(680,0),(680,1000),(0,255,0),2)
            cv2.line(frame,(712,0),(712,1000),(0,255,0),2)
        elif cam_id=="481": # Horizontal
              cv2.line(frame,(3,380),(1018,380),(0,255,0),2)
              cv2.line(frame,(5,400),(1019,400),(0,255,255),2)
        elif cam_id=="482": # Horizontal
            cv2.line(frame,(520,0),(520,1000),(0,255,0),2)
            cv2.line(frame,(560,0),(560,1000),(0,255,0),2)
        elif cam_id=="487": #Vertical
            cv2.line(frame,(520,0),(520,1000),(0,255,0),2)
            cv2.line(frame,(560,0),(560,1000),(0,255,0),2)


    
        down=len(counter1)
        up=len(counter2)
        cvzone.putTextRect(frame,f'Entry  {down}',(50,60) ,2,2)
        cvzone.putTextRect(frame,f'Exit    {up}'  ,(50,160),2,2)
        # print(cam_id,down,up)
        if count%10==0:
            insert_data_query = """
            INSERT INTO Entry_Exit_Employees (class1, camera_code, camera_unit, date1, time1,area, entry, exit)
            VALUES (%s, %s, %s, %s, %s,%s,%s,%s);
            """
            current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # db_cursor.execute(insert_data_query, ('person', cam_id,ip,current_datetime, current_datetime,area,down,up))
            # db_connection.commit()
        cv2.imshow(f'RGB{cam_id}',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        # db_cursor.close()
        # db_connection.close()