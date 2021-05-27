import cv2
import numpy as np
from time import sleep



min_width = 80  # Minimum rectangle width 
min_height = 80  # Minimum rectangle height 
allowable_range = 6  # Allowable error between pixel 
line_pos = 100  # Count line position   
delay = 25  # Video FPS
detect = []
def centroid(x, y, w, h): #centroid of coordinates
    x1 = w // 2
    y1 = h // 2
    cx = x + x1
    cy = y + y1
    return cx, cy


def set_info(detect):
    global vehicles      
    for (x, y) in detect:
        if (line_pos + allowable_range) > y > (line_pos - allowable_range):
            vehicles += 1
            cv2.line(frame1, (0, line_pos), (1920, line_pos), (0, 0, 255), 3)
            detect.remove((x, y))
            print("Vehicles detected so far : " + str(vehicles))


def show_info(frame1, dilated):
    text = f'Total vehicles: {vehicles}'
    cv2.putText(frame1, text, (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame1)
    cv2.imshow("Ditector ", dilated)


vehicles = 0 
cap = cv2.VideoCapture('video.mp4')
#subtract = cv2.bgsegm.createBackgroundSubtractorMOG()  # Take the bottom and subtract from what is moving
subtract =cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)
while True:
    ret, frame1 = cap.read()  # Take each frame of the video
    frame1=frame1[320:1080,560:1920]
    #cv2.imshow("frame",frame)
    time = float(1 / 30)
    sleep(time)  # Delays between each processing
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Take the frame and transform it to black and white
    
    #cv2.imshow("GREY",grey)
    
    blur = cv2.GaussianBlur(grey, (3, 3), 5)  # Blur to try to remove imperfections from the image
    #cv2.imshow("blur",blur)
    
    
    img_sub = subtract.apply(blur)  # Subtracts the image applied in the blur
    #cv2.imshow("img_sub",img_sub)
    

    
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))  # "Thicken" what's left of the subtraction
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Creates a 5x5 matrix, where the matrix format between 0 and 1 forms an ellipse inside
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)  # Try to fill all the "holes" in the image
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    #_,dilated =cv2.threshold(dilated,254,255,cv2.THRESH_BINARY)

    contour, img = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (0, line_pos), (1920, line_pos), (255, 127, 0), 3)
    for (i, c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        valid_contour = (w >= min_width) and (h >= min_height)
        if not valid_contour:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centro = centroid(x, y, w, h)
        detect.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

    set_info(detect)
    show_info(frame1, dilated)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
