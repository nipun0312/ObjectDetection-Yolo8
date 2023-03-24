from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture('Videos/bikes.mp4')
cap.set(3,1280)
cap.set(4,720)

model = YOLO("../YOLO-weight/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            ###Bounding Box
            ###For OpenCV
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (w, h), (255,0,255),2)

            ###For cvzone
            w, h = x2-x1, y2-y1
            # bbox = int(x1), int(y1), int(w), int(h)
            cvzone.cornerRect(img, (x1, y1, w, h))

            ###Confidence
            conf = math.ceil((box.conf[0]*100))/100

            ###Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1, font=cv2.FONT_HERSHEY_COMPLEX, offset=5, colorB=(0, 255, 0))


    cv2.imshow('Image', img)
    cv2.waitKey(1)

