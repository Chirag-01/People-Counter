import cv2
from ultralytics import YOLO
import os
import threading
import time
# import boto3
import psycopg2 as spg
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime
import json
import torch
from main_final import track_person
 
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
 
def create_table_if_not_exists(cur):

    create_table_sql = """

    CREATE TABLE IF NOT EXISTS real_time_person_count (

        id SERIAL PRIMARY KEY,

        class1 TEXT,

        production_house TEXT,

        camera_unit TEXT,

        date1 date,

        time1 time,

        area TEXT,

        person_count INT
    );

    """

    cur.execute(create_table_sql)
 
class DetectTray:
 
    def _init_(self):

        self.device = torch.device(0)
        print(self.device)

        self.model = YOLO("yolov8m.pt").to(self.device)
 
    def draw_boxes(self, image, bbox, class_label):

        x, y, width, height = bbox

        color = (0, 255, 0)

        cv2.rectangle(image, (x - width // 2, y - height // 2), (x + width // 2, y + height // 2), color, 2)

        cv2.putText(image, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image
 
    def process_camera(self, stream_data, plant, production_house, area_type, ip_address, ppe_list):

        try:

            con, cur = connect_to_db()

            if con is None or cur is None:

                print("Failed to connect to the database.")

                return
 
            create_table_if_not_exists(cur)
 
            stream_link = stream_data['streamLink']
 
            try:

                front = cv2.VideoCapture(stream_link)

            except cv2.error as cv2_error:

                print(f"Error opening video stream {stream_link}: {cv2_error}")

                return
 
            frame_number = 0

            last_detection_time = time.time()
 
            if not front.isOpened():

                print(f"Error: Video stream not opened for {stream_link}.")

                return
 
            while True:

                try:

                    ret, frame = front.read()

                    if not ret:

                        print("Error reading frame: ret =", ret)

                        time.sleep(3)

                        continue
 
                    current_time = time.time()

                    if current_time - last_detection_time < 45:

                        continue

                    last_detection_time = current_time
 
                    frame = cv2.resize(frame, (720, 620))

                    names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
                    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
                    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
                    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 
                    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
                    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
                    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 
                    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
 
                    predictions = self.model.predict(frame, conf=0.40)

                    box = {}
 
                    for result in predictions:

                        boxes = result.boxes

                        xywh = boxes.xywh

                        classes = boxes.cls
 
                        for b, c in zip(xywh, classes):

                            b = b.tolist()

                            b = [round(num) for num in b]

                            x = int(b[0])

                            catch = names[int(c)]
                            tensor_list = classes.tolist()
                            count_of_person = tensor_list.count(0.0)
                            print(count_of_person)
                            text = f"Count of person: {count_of_person}"
 
                            if any(char.isupper() for char in catch):

                                box[x] = catch
 
                            labeled_image = self.draw_boxes(frame, b, catch)

                            date1 = datetime.now().strftime("%Y-%m-%d")

                            time1 = datetime.now().strftime("%H:%M:%S")

                            now = datetime.now()

                            time_1 = now.strftime("%H%M%S")

                            date = now.strftime("%Y%m%d")

                            im_id = str(str(date1) + str(time_1) +"_"+ str(production_house) + str(area_type))

                            frame_to_upload = cv2.resize(labeled_image, (1920, 1080), cv2.INTER_CUBIC)
                            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            org = (50, 50)
                            fontScale = 1
                            color = (255, 0, 0)
                            thickness = 2
                            image = cv2.putText(frame_to_upload, text, org, font,fontScale, color, thickness, cv2.LINE_AA) 
                            cv2.imwrite("output_images/detected" + str(production_house) + str(timestamp) + ".jpg", frame_to_upload)
 
                            image_data = cv2.imencode('.jpg', frame_to_upload)[1].tobytes()
 
                            object_name = str(im_id) + '.jpg'

                            print(production_house)

                            print(area_type)
 
                            with con, con.cursor() as cur_thread:

                                if catch in ppe_list:

                                    try:

                                        cur_thread.execute("""

                                            INSERT INTO real_time_person_count (class1, production_house, camera_unit, date1, time1, area, person_count)

                                            VALUES(%s, %s, %s, %s, %s, %s, %s);

                                        """, (catch, production_house, ip_address, date1, time1,area_type,count_of_person))

                                        con.commit()

                                        print(f"Detected: {catch}")

                                    except spg.Error as e:

                                        con.rollback()

                                        print("Error inserting data into the database:", str(e))

                                        # You can log the error or take further action if needed

                                else:

                                    print(f"Not saving data for {catch}")
 
                    frame_number += 1
 
                except cv2.error as cv2_error:

                    print("OpenCV Error: ", cv2_error)

                except Exception as e:

                    print("An error occurred:", str(e))

                finally:

                    time.sleep(0.1)  # Optional: Add a small delay to reduce CPU usage

        finally:

            con.close()  # Close the connection when the thread is done
 
if __name__ == '__main__':

    # devices = json.loads(open('./camera.json').read())
    devices = json.loads(open('./camera.json').read())

    threads = []

 
    for plant, plantData in devices.items():

        for productionHouse, productionHouseData in plantData.items():

            for areaType, areaData in productionHouseData.items():

#                 stream_link = areaData['streamLink']


#                 after_at = stream_link.split('@')[1]
# #
#                 ip_address = after_at.split(':')[0]
 
#                 ppe_list = areaData.get("ppeList", [])


 
 
                thread = threading.Thread(target=track_person,

                                          args=(areaData,productionHouse,"",""))

                # thread = threading.Thread(target=track_person,

                #                           args=(areaData,productionHouse,ip_address,areaType))

                threads.append(thread)
 
    for thread in threads:

        thread.start()
 
    for thread in threads:

        thread.join()